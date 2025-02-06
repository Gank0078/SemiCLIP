import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
from torch.nn import functional as F
from open_clip import get_tokenizer
from timm.models.vision_transformer import Block, PatchEmbed
from functools import partial
from custom.pos_embed import get_2d_sincos_pos_embed


def create_custom_model(args, model, device):
    return peftCLIP(args, model, device)


class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)

    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


def text_global_pool(x, text=None, pool_idx=None, pool_type='argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if pool_idx is not None:
            pooled, tokens = x[torch.arange(x.shape[0]), pool_idx], x
        else:
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class peftCLIP(nn.Module):
    override = ["clip", "forward", "lock_text_tower", "logit_scale_con"]

    def __init__(self, args, clip, device):
        super().__init__()
        self.clip = clip
        self.dtype = clip.visual.proj.dtype
        self.loss_type = args.loss_type

        # For visual model
        self.conv1 = clip.visual.conv1
        self.class_embedding = clip.visual.class_embedding
        self.vision_positional_embedding = clip.visual.positional_embedding
        self.ln_pre = clip.visual.ln_pre
        self.vision_transformer = clip.visual.transformer
        self.ln_post = clip.visual.ln_post
        self.proj = clip.visual.proj
        self.logit_scale = clip.logit_scale
        self.output_dict = True

        # For text model
        self.token_embedding = clip.token_embedding
        self.text_positional_embedding = clip.positional_embedding
        self.text_transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        self.attn_mask = clip.attn_mask
        self.text_pool_type = clip.text_pool_type

        self.stage = args.stage
        self.plan = args.plan
        self.layer = args.layer
        self.tokenizer = get_tokenizer(args.model)

        # peft related variable
        n_layers = clip.visual.transformer.layers
        emb_dim = clip.visual.transformer.width
        seq_len = clip.visual.positional_embedding.shape[0]
        patch_size = clip.visual.conv1.kernel_size

        self.VPT = args.VPT
        self.peft_vpt_list = [None] * n_layers
        self.method = args.method
        self.th = args.th
        self.det = args.det

        if args.method.startswith("semiclip"):
            self.nouns_logit_scale = nn.Parameter(torch.ones([]).to(device) * self.logit_scale.item())
            if self.stage > 1:
                # self.words = args.words
                self.words = 4
                self.prompts_xs = "a photo includes X a photo includes X a photo includes X a photo includes X"
                self.prompts_xs = args.tokenizer(self.prompts_xs).to(device)
                self.pre_idx = [1, 2, 3] + [5, 6, 7] + [9, 10, 11] + [13, 14, 15]
                self.xs_idx = torch.where(self.prompts_xs[0] == 343)[0].tolist()
                initw = self.token_embedding(self.prompts_xs[0]).data
                initw = initw[1:4]
                self.nouns_xs = nn.Parameter(torch.empty(3 * self.words, 512).to(device))
                self.nouns_xs.data = initw.repeat(self.words, 1).clone()

    def lock_text_tower(self):
        # ignore options and just lock the entire text tower
        for param in self.clip.transformer.parameters():
            param.requires_grad = False

    def lock_vision_tower(self):
        # ignore options and just lock the entire vision tower
        for name, param in self.named_parameters():
            if name.startswith("clip.visual"):
                param.requires_grad = False

    def lock_module_tower(self, frozen_list):
        for name, param in self.named_parameters():
            if name in frozen_list:
                param.requires_grad = False

    def encode_peft_image(self, x, normalize=True, hidden_states=False):
        # x = self.conv1(x).float()  # shape = [*, width, grid, grid]
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vision_positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = self.vision_transformer.layers

        for i in range(n_layers):
            block = self.vision_transformer.resblocks[i]

            vpt = self.peft_vpt_list[i]

            if vpt is not None:
                x = vpt(x)

            _seq_len_after_vpt = x.shape[1]

            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj = _mlp[0]
            _mlp_gelu = _mlp[1]
            _mlp_out_proj = _mlp[2]

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads
            residual = x  # deep copy

            x = _ln_1(x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)

            q = q / math.sqrt(_head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            x = torch.bmm(attn, v)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)

            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)
            x = residual + x
            residual = x  # deep copy

            x = _ln_2(x)
            x = _mlp_in_proj(x)
            x = _mlp_gelu(x)
            x = _mlp_out_proj(x)

            x = residual + x

            x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = x
        hidden = self.ln_post(hidden)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        if hidden_states:
            return x, hidden
        else:
            return x

    def encode_peft_text(self, text, normalize=True, pseu_words=None, xs=None):
        cast_dtype = self.text_transformer.get_cast_dtype()
        pool_idx = None
        if xs is not None:
            x = xs
        else:
            x = self.token_embedding(text).to(cast_dtype)

        x = x + self.text_positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x, _ = text_global_pool(x, text, pool_idx, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, image, text, query=None, qtext=None, keyword=None, image1=None, query1=None,
                qidxs=None, nidxs=None, norm=True, qlogits=None, rtext=None, nlabels=None, nouns=False, logits_w=None):
        image_features = self.encode_peft_image(image, normalize=norm)
        text_features = self.encode_peft_text(text, normalize=norm)
        out = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.clip.logit_scale.exp()
        }

        if image1 is not None:
            image_features1 = self.encode_peft_image(image1, normalize=norm)
            out.update({
                "image_features1": image_features1,
            })

        if query is not None:  # unlabeled image
            query_features = self.encode_peft_image(query, normalize=norm)
            out.update({
                "query_features": query_features,
            })

        if query1 is not None:  # unlabeled image
            query_features1 = self.encode_peft_image(query1, normalize=norm)
            out.update({
                "query_features1": query_features1,
            })

        if qtext is not None:
            qtext_features = self.encode_peft_text(qtext, normalize=True)

            out.update({
                "qtext_features": qtext_features,
            })

        if nouns is True:
            logits_l = self.nouns_logit_scale.exp() * image_features @ self.nouns_emb.T
            out.update({
                "logits_l": logits_l,
            })
            if query1 is not None:
                logits_s = self.nouns_logit_scale.exp() * query_features1 @ self.nouns_emb.T
                out.update({
                    "logits_s": logits_s,
                })
            if logits_w is not None:
                _, topk_idx = torch.topk(logits_w.detach(), self.words, dim=1)
                prompts_xs = self.prompts_xs.repeat(len(query_features), 1)
                prompts_xs[:, self.xs_idx] = self.nouns_tokens[topk_idx]
                ptulab = self.token_embedding(prompts_xs)
                ptulab[:, self.pre_idx] = self.nouns_xs
                qtext_pseu_features = self.encode_peft_text(prompts_xs, normalize=True, xs=ptulab)
                out.update({
                    "qtext_pseu_features": qtext_pseu_features,
                })
        if self.output_dict:
            return out

        return image_features, text_features, self.logit_scale.exp()
