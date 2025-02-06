import time
import math
import logging

import torch

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_cast_dtype
from training.precision import get_autocast
from training.distributed import is_master
from training.train import AverageMeter, backward, unwrap_model
from itertools import cycle
from open_clip import get_tokenizer
import numpy as np


def get_nouns_emb(nouns, model, device, tokenizer):
    # temp = "a photo includes {}."
    temp = "a photo includes {}."
    # temp = "a photo of {}."
    prompts = [temp.format(c.replace("_", " ")) for c in nouns]

    prompts = torch.cat([tokenizer(p) for p in prompts])
    prompts = prompts.to(device)

    with torch.no_grad():
        text_features = model.encode_peft_text(prompts, normalize=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    # num_batches_per_epoch = args.steps
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    data['query'].set_epoch(epoch)
    dataloader_query = iter(data['query'].dataloader)

    # if args.update_nouns:
    #     nouns_emb = get_nouns_emb(data['nouns'], model, device, tokenizer=get_tokenizer(args.model))
    #     model.nouns_emb.data = nouns_emb

    norm = True
    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        qlabels = None
        labels = None
        nidxs = None
        nreal = None
        qtext = None
        qreal = None
        if args.method.startswith("semiclip"):
            images, texts, nidxs, labels = batch[0], batch[1], batch[2], batch[3]
            nouns = True
        else:
            images, texts, nidxs = batch[0], batch[1], batch[2]
            nouns = False
        texts = texts.to(device=device, non_blocking=True)

        if labels is not None:
            labels = labels.to(device=device, non_blocking=True)
        if isinstance(images, list):
            images = images[0].to(device=device, dtype=cast_dtype, non_blocking=True)
            inputs = {'image': images, 'text': texts}
        else:
            images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
            inputs = {'image': images, 'text': texts}

        model_out_extra = {}
        qbatch = next(dataloader_query)
        if args.train_data == "Simpsons" or args.train_data == "RS-SHIFT":
            queries, qidxs = qbatch[0], qbatch[1]
        else:
            queries, qtext, qidxs = qbatch[0], qbatch[1], qbatch[2]
            qreal = qbatch[-1]
            nreal = batch[-1]

        if args.stage == 1:
            if isinstance(queries, list):
                queries = queries[0].to(device=device, dtype=cast_dtype, non_blocking=True)
            else:
                queries = queries.to(device=device, dtype=cast_dtype, non_blocking=True)
        elif args.stage == 2:
            queries, queries1 = queries[0], queries[1]
            queries = queries.to(device=device, dtype=cast_dtype, non_blocking=True)
            queries1 = queries1.to(device=device, dtype=cast_dtype, non_blocking=True)
            inputs.update({'query1': queries1})
        else:
            queries = queries.to(device=device, dtype=cast_dtype, non_blocking=True)
        inputs.update({'query': queries})
        inputs.update({'nouns': nouns})
        if qtext is not None:
            qtext = qtext.to(device=device, non_blocking=True)
            inputs.update({'qtext': qtext})

        if qidxs is not None:
            qidxs = qidxs.to(device=device, non_blocking=True)
            inputs.update({'qidxs': qidxs})
        if labels is not None:
            inputs.update({'nlabels': labels})

        inputs.update({'norm': norm})

        logits_w = None
        if args.method.startswith("semiclip") and qidxs is not None and args.idxs2logits is not None:
            logits_w = [args.idxs2logits[idx] for idx in qidxs.tolist()]
            logits_w = np.stack(logits_w)
            logits_w = torch.tensor(logits_w).to(device)
            inputs.update({'logits_w': logits_w})

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(**inputs)
                model_out.update(model_out_extra)  # extra info for loss
                logit_scale = model_out["logit_scale"]

                if labels is not None:
                    model_out.update({'nlabels': labels})
                if nidxs is not None:
                    nidxs = nidxs.to(device=device, non_blocking=True)
                    model_out.update({'nidxs': nidxs})
                if qidxs is not None:
                    qidxs = qidxs.to(device=device, non_blocking=True)
                    model_out.update({'qidxs': qidxs})
                if qreal is not None:
                    model_out.update({'qreal': qreal})
                if nreal is not None:
                    model_out.update({'nreal': nreal})
                if logits_w is not None:
                    model_out.update({'logits_w': logits_w})
                if qtext is not None:
                    qtext = qtext.to(device=device, non_blocking=True)
                    model_out.update({'qtext': qtext})
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}' : v for k, v in dist_model_out.items()})

                losses, addinfo = loss(**model_out, output_dict=True, model=model)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(**inputs)
                    model_out.update(model_out_extra)  # extra info for loss
                    model_out.pop("logit_scale")
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
                    losses, _ = loss(**inputs, logit_scale=logit_scale, output_dict=True, model=model)
                    del inputs
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        fg = 0
        if args.method == "nouns_ssl_log" or args.method == "ours_log":
            fg = 1
            log_data = {
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.item() for name, val in addinfo.items()})

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            # log_data = {
            #     "lr": optimizer.param_groups[0]["lr"]
            # }
            if fg == 0:
                log_data = {
                    "lr": optimizer.param_groups[0]["lr"]
                }
                log_data.update({name: val.item() for name, val in addinfo.items()})
            # log_data.update({name: val.item() for name, val in addinfo.items()})
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
