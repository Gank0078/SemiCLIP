import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown, cosine_lr_multi_group
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync

# use custom functions
from custom.params import parse_args
from custom.data import get_data
from custom.loss import create_loss
from custom.train import train_one_epoch
from custom.model import create_custom_model
from custom.evaluate import evaluate
from torch.utils.data import DataLoader, TensorDataset
import pickle


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def topk_acc(logits, labels, k=3):
    _, topk_indices = torch.topk(logits.detach(), k, dim=1)
    one_hots = torch.zeros_like(logits.detach())
    one_hots.scatter_(1, topk_indices, 1)
    nouns_acc = (one_hots * labels).sum(dim=1).mean()
    return nouns_acc


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def get_nouns_emb(args, nouns, model, device, tokenizer):
    temp = "a photo includes {}."
    prompts = [temp.format(c.replace("_", " ")) for c in nouns]

    prompts = torch.cat([tokenizer(p) for p in prompts])
    prompts = prompts.to(device)

    with torch.no_grad():
        text_features = model.encode_peft_text(prompts, normalize=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def main(args):
    args = parse_args(args)

    # args.stage = 1
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        keyword_type = args.keyword_path.split('/')[-1].split('.')[0]\
            if args.keyword_path is not None else 'none'
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"data_{args.train_data}",
            f"ratio_{args.label_ratio}",
            f"model_{model_name_safe}",
            f"method_{args.method}",
            f"keyword_{keyword_type}",
            f"seed_{args.seed}",
        ])

    resume_latest = args.resume == 'latest'
    # log_base_path = os.path.join(args.logs, args.name)
    log_base_path = args.logs
    args.log_path = None
    if args.train_data and is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(args.log_path) and not resume_latest:
        #     print(
        #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
        #     )
        #     return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if args.train_data and is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    # if resume_latest:
    if args.resume_path is not None:
        resume_from = None
        # checkpoint_path = args.checkpoint_path
        checkpoint_path = args.resume_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )
    if args.distill:
        # FIXME: currenlty assumes the model your distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )

    args.tokenizer = get_tokenizer(args.model)
    model = create_custom_model(args, model, device)  # use custom model
    args.idxs2logits = None
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    start_epoch = 0
    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'

    args.n_cls = None
    if args.method.startswith('semiclip') and data['nouns'] is not None:
        nouns_emb = get_nouns_emb(args, data['nouns'], model, device, tokenizer = get_tokenizer(args.model))
        model.nouns_emb = nn.Parameter(nouns_emb)
        args.n_cls = len(data['nouns'])
        args.nouns = data['nouns']
        args.tokenizer = get_tokenizer(args.model)

        nouns_tokens = []
        for word in args.nouns:
            nouns_tokens.append(args.tokenizer(word)[0, 1].unsqueeze(0))
        model.nouns_tokens = torch.cat(nouns_tokens).to(device)
    else:
        args.nouns = None

    if args.train_data and is_master(args):
        # logging.info("Model:")
        # logging.info(f"{str(model)}")
        logging.info("Params:")
        # params_file = os.path.join(args.logs, args.name, "params.txt")
        params_file = os.path.join(log_base_path, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        # find_unused_parameters = True
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data:
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        peft_module = lambda n, p: n.startswith('peft')
        nouns_module = lambda n, p: n.startswith('nouns')

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and not nouns_module(n, p) and
                               not peft_module(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and not nouns_module(n, p) and
                       not peft_module(n, p) and p.requires_grad]

        nouns_params = [p for n, p in named_parameters if nouns_module(n, p) and p.requires_grad]
        peft_params = [p for n, p in named_parameters if peft_module(n, p) and p.requires_grad]
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0., "name": "base"},
                {"params": rest_params, "weight_decay": args.wd, "name": "base"},
                {"params": nouns_params, "lr": args.lr_add, "weight_decay": args.wd, "name": "add"},
                {"params": peft_params, "lr": args.lr_add, "weight_decay": args.wd, "name": "add"},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0

    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            msg = model.load_state_dict(sd, strict=False)
            logging.info(f"=> resuming checkpoint '{args.resume}'")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # create scheduler if train
    scheduler = None
    if args.train_data and optimizer is not None:
        # total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.steps is not None:
            total_steps = args.steps * args.epochs
        else:
            total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            # scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
            scheduler = cosine_lr_multi_group(optimizer, [args.lr, args.lr_add], args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            # cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            if args.steps is not None:
                cooldown_steps = args.steps * args.epochs_cooldown
            else:
                cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.train_data and args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if not args.train_data:
        metrics = evaluate(model, data, start_epoch, args, writer)
        with open('eval.txt', 'a') as f:
            for k, v in metrics.items():
                if k == "zeroshot-val-top1":
                    f.write('{}\t{}\t{}\t{:.2f}\n'.format(
                        args.name, args.imagenet_val, k, 100 * v))
                elif k in ["image_to_text_R@1", "image_to_text_R@5", "image_to_text_R@10",
                           "text_to_image_R@1", "text_to_image_R@5", "text_to_image_R@10"]:
                    f.write('{}\t{}\t{}\t{:.2f}\n'.format(
                        args.name, args.val_data, k, 100 * v))
        return

    idxs2logits = None
    if args.pkname is not None and args.stage == 2:
        with open('../results/sclip_load/' + args.pkname + '.pkl', 'rb') as f:
            idxs2logits = pickle.load(f)
    args.idxs2logits = idxs2logits
    model.idxs2logits = idxs2logits
    loss = create_loss(args, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params}')
    tuned_params = 0
    for p in model.parameters():
        if p.requires_grad:
            tuned_params += p.numel()
    print(f'Tuned params: {tuned_params}')

    if args.only_val:
        evaluate(model, data, 0, args, writer)
    else:
        for epoch in range(start_epoch, args.epochs):
            if is_master(args):
                logging.info(f'Start epoch {epoch}')

            train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
            completed_epoch = epoch + 1

            if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
                if args.eval_last:
                    if completed_epoch == args.epochs:
                        evaluate(model, data, completed_epoch, args, writer)
                else:
                    evaluate(model, data, completed_epoch, args, writer)

            # Saving checkpoints.
            if args.save_logs:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                if args.save_ckpt and completed_epoch == args.epochs:
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                    )

        if args.pkname is not None and args.stage == 1:
            model.eval()
            dataloader = DataLoader(
                data['query'].dataloader.dataset,
                batch_size=32,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            )
            all_reps = []
            all_idxs = []
            for idx, batch in enumerate(dataloader):
                if args.train_data == "Simpsons" or args.train_data == "RS-SHIFT":
                    images, idxs = batch[0], batch[1]
                else:
                    images, texts, idxs = batch[0], batch[1], batch[2]

                if isinstance(images, list):
                    images, idxs = images[0].to(device), idxs.to(device)  # 选择弱数据增强的
                else:
                    images, idxs = images.to(device), idxs.to(device)

                if args.distributed and not args.horovod:
                    image_features = model.module.encode_peft_image(images, normalize=True)
                    reps = model.module.nouns_logit_scale.exp() * image_features @ model.module.nouns_emb.T
                else:
                    image_features = model.encode_peft_image(images, normalize=True)
                    reps = model.nouns_logit_scale.exp() * image_features @ model.nouns_emb.T
                reps = [x.detach().cpu().numpy() for x in reps]
                idxs = [x.detach().cpu().numpy() for x in idxs]
                all_reps.extend(reps)
                all_idxs.extend(idxs)
            all_reps = np.stack(all_reps)
            all_idxs = np.stack(all_idxs)

            idxs2logits = {index: all_reps[i] for i, index in enumerate(all_idxs)}

            with open('../results/sclip_load/' + args.pkname + '.pkl', 'wb') as f:
                pickle.dump(idxs2logits, f)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
