import torch
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import wandb
except ImportError:
    wandb = None

from open_clip.loss import ClipLoss
import ot
# from ranking import TrueRanker, rank_normalised
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import choice
# from textaugment import EDA

cc = 0


def entropy(x, mean=False):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ = torch.clamp(x, min = EPS)
    b = x_ * torch.log(x_)

    if len(b.size()) == 2:  # Sample-wise entropy
        if mean is True:
            return - b.sum(dim=1).mean()
        else:
            return - b.sum(dim=1)
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


def create_loss(args, device):
    return SemiSupervisedClipLoss(
        args.method,
        args=args,
        device=device,
        pseudo_label_type=args.pseudo_label_type,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


class SemiSupervisedClipLoss(ClipLoss):
    def __init__(
            self,
            method,
            args=None,
            device=None,
            pseudo_label_type="ot-image",
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.method = method
        self.pseudo_label_type = pseudo_label_type
        self.wcon = args.wcon
        self.det = args.det
        self.cylam1 = args.cylam1
        self.cylam2 = args.cylam2
        self.cylam3 = args.cylam3
        self.cylam4 = args.cylam4
        self.cylam5 = args.cylam5
        self.cylam6 = args.cylam6
        self.cylam7 = args.cylam7
        self.Tr = args.Tr
        self.ranklam = args.ranklam
        self.Tcache = args.Tcache
        self.qlen = args.qlen
        self.full_flag = False
        self.th = args.th
        self.smin = args.smin
        self.smax = args.smax

        self.distributed = args.distributed
        self.horovod = args.horovod
        self.n_cls = args.n_cls
        self.relu = nn.ReLU()
        self.mse = torch.nn.MSELoss()
        self.topk = args.topk
        self.maskmode = args.maskmode
        self.lossmode = args.lossmode
        self.alpha = args.alpha
        self.idxs2logits = args.idxs2logits
        self.Ts = args.Ts
        self.wnouns = args.wnouns
        self.device = device
        self.pratio = args.pratio
        self.stage = args.stage

        self.prompts = "a photo includes {}."
        self.nouns = args.nouns
        if self.nouns is not None:
            self.tokenizer = args.tokenizer
            self.words = args.words
            self.tau = args.tau
            self.selk = args.selk

    def topk_acc(self, logits, labels, k=3):
        _, topk_indices = torch.topk(logits.detach(), k, dim=1)
        one_hots = torch.zeros_like(logits.detach())
        one_hots.scatter_(1, topk_indices, 1)
        # nouns_acc = (one_hots * labels).sum(dim=1).mean()
        nouns_acc = (one_hots * labels).sum() / (len(logits) * k)
        return nouns_acc

    def topk_acc_u(self, logits, texts, tokens, k=3):
        _, topk_indices = torch.topk(logits.detach(), k, dim=1)
        topk_tokens = tokens[topk_indices]
        cc = 0
        for i in range(len(logits)):
            for j in range(k):
                if (topk_tokens[i, j] == texts[i]).any():
                    cc += 1
        nouns_acc = torch.tensor([cc / (len(logits) * k)]).to(self.device)
        return nouns_acc

    def forward(self, image_features, text_features=None, logit_scale=None, output_dict=False,
                query_features=None, qtext_features=None, cap_features=None, keyword_features=None,
                keyword_labels=None, loss=None, image_features1=None, query_features1=None, model=None,
                nlabels=None, qlabels=None, qidxs=None, text_map_features=None, qtext_map_features=None,
                pseu_words=None, query_pseu_words=None, nidxs=None, qtext_map_features1=None, nreal=None, qreal=None,
                logits_l=None, logits_w=None, logits_s=None, qtext_pseu_features=None, pseusim=None,
                qsel_features=None, qtext=None):
        device = image_features.device
        losses = dict()  # dict of losses
        addinfo = dict()
        bce = nn.BCELoss()
        bce_logits = nn.BCEWithLogitsLoss()

        bs = image_features.shape[0]

        # gather tensors over worlds
        if self.world_size > 1:
            dist_kwargs = {
                "local_loss": self.local_loss,
                "gather_with_grad": self.gather_with_grad,
                "rank": self.rank,
                "world_size": self.world_size,
                "use_horovod": self.use_horovod,
            }
            image_features = gather_features(image_features, **dist_kwargs)
            text_features = gather_features(text_features, **dist_kwargs)

            query_features = gather_features(query_features, **dist_kwargs)
            keyword_labels = gather_features(keyword_labels, **dist_kwargs)
            if self.method.startswith('semiclip'):
                nlabels = gather_features(nlabels, **dist_kwargs)
                logits_l = gather_features(logits_l, **dist_kwargs)
                logits_w = gather_features(logits_w, **dist_kwargs)
                logits_s = gather_features(logits_s, **dist_kwargs)
                qtext_pseu_features = gather_features(qtext_pseu_features, **dist_kwargs)

        # compute loss
        if self.method == "base":
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T

            labels = self.get_ground_truth(device, image_features.shape[0])
            losses["contrastive_loss"] = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2 * self.wcon

        elif self.method == "semiclip" and self.stage == 1:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T
            labels = self.get_ground_truth(device, image_features.shape[0])
            losses["contrastive_loss"] = (
                                                 F.cross_entropy(logits_per_image, labels) +
                                                 F.cross_entropy(logits_per_text, labels)
                                         ) / 2 * self.wcon

            if logits_l is not None:
                nlabels = ((nlabels) / (nlabels + 1e-8).sum(dim=1).unsqueeze(1))
                losses["soft_loss"] = soft_cross_entropy(logits_l / self.Tr, nlabels) * self.cylam1

        elif self.method == "semiclip" and self.stage == 2:
            bs = image_features.shape[0]
            ubs = query_features.shape[0]
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T
            labels = self.get_ground_truth(device, bs)
            # labels_u = self.get_ground_truth(device, query_features.shape[0])
            losses["contrastive_loss"] = (
                                                 F.cross_entropy(logits_per_image, labels) +
                                                 F.cross_entropy(logits_per_text, labels)
                                         ) / 2 * self.wcon

            probs_w = F.softmax(logits_w, dim=1)
            pseu_probs, pseu_idxs = torch.topk(probs_w.detach(), self.topk, dim=1)
            pseu_ot = F.one_hot(pseu_idxs, num_classes=logits_s.shape[1])
            pseu_ot = pseu_ot.sum(dim=1)
            pseu_ot = (pseu_ot / (pseu_ot + 1e-8).sum(dim=1).unsqueeze(1))
            losses["softu_loss"] = soft_cross_entropy(logits_s / self.Tr, pseu_ot) * self.cylam1

            sim = (qtext_pseu_features * query_features).sum(dim=1)
            selk = int(ubs * self.pratio)
            _, topkpos = torch.topk(sim, k=selk)
            mask = torch.zeros_like(sim, dtype=torch.bool)
            mask[topkpos] = True
            # addinfo["qtext_sim"] = (qtext_pseu_features * qtext_features).sum(dim=1).mean()
            addinfo["mask_num"] = torch.sum(mask)

            image_mix = torch.cat((image_features, query_features[mask]))
            text_mix = torch.cat((text_features, qtext_pseu_features[mask]))
            logits_image_per_image = logit_scale * image_mix @ image_mix.t()
            logits_text_per_text = logit_scale * text_mix @ text_mix.t()
            losses["inmodal_cyclic_loss"] = (logits_image_per_image - logits_text_per_text).square().mean() / (
                    logit_scale * logit_scale) * bs * self.cylam2
            logits_per_image1 = logit_scale * image_mix @ text_mix.T
            logits_per_text1 = logits_per_image1.T
            losses["crossmodal_cyclic_loss"] = (logits_per_image1 - logits_per_text1).square().mean() / (
                    logit_scale * logit_scale) * bs * self.cylam3

        if output_dict:
            return losses, addinfo
        else:
            return sum(losses.items()), addinfo
        # return losses if output_dict else sum(losses.items())


def get_assignments(query, image, text, logit_scale, pseudo_label_type):
    if pseudo_label_type == "hard-image":
        plan = hard_nn(query, image)
    elif pseudo_label_type == "hard-text":
        plan = hard_nn(query, image)
    elif pseudo_label_type == "soft-image":
        plan = soft_nn(query, image, logit_scale)
    elif pseudo_label_type == "soft-text":
        plan = soft_nn(query, text, logit_scale)
    elif pseudo_label_type == "ot-image":
        plan = ot_plan(query, image, logit_scale)
    elif pseudo_label_type == "ot-text":
        plan = ot_plan(query, image, logit_scale)
    else:
        raise NotImplementedError
    return plan

def hard_nn(query, support):
    _, idx = (query @ support.T).max(dim=1)
    plan = F.one_hot(idx, len(support)).float()
    return plan


def soft_nn(query, support, logit_scale):
    plan = F.softmax(query @ support.T * logit_scale, dim=1)
    return plan


def ot_plan(query, support, logit_scale):
    global cc
    C = 1 - query @ support.T  # (query, batch)
    reg = 1 / logit_scale  # learned temperature

    dim_p, dim_q = C.shape
    p = torch.ones(dim_p, device=C.device, dtype=torch.double) / dim_p
    q = torch.ones(dim_q, device=C.device, dtype=torch.double) / dim_q
    cc = cc + 1
    # print(cc)
    # print((p.dtype, q.dtype, C.dtype, reg.dtype))
    # if cc == 29:
    #     print('-')
    # P = ot.bregman.sinkhorn(p, q, C, reg=reg, numItermax=10)
    P = ot.bregman.sinkhorn(p, q, C.double(), reg=reg, numItermax=10)

    plan = P / P.sum(dim=1, keepdim=True)
    plan = plan.type_as(support)
    return plan


def soft_cross_entropy(outputs, targets, weight=1.):
    loss = -targets * F.log_softmax(outputs, dim=1)
    return (loss * weight).sum(dim=1).mean()
    # return (loss.sum(dim=1) * weight).mean()


def gather_features(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    if features is None:
        return features

    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        else:
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features[rank] = features
            all_features = torch.cat(gathered_features, dim=0)

    return all_features
