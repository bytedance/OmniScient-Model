"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/train.py
"""

import argparse
import glob
import os
import random

import numpy as np
import torch
from data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from train_utils import (
    train_one_epoch,
    get_mp_policy_dtype,
    save_checkpoint,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
import functools

from modeling.factory import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument("--base_model_name", default="Salesforce/instructblip-vicuna-7b", type=str)
    parser.add_argument("--input_size", default=1120, type=int)
    parser.add_argument("--random_scale_min", default=0.5, type=float)
    parser.add_argument("--random_scale_max", default=1.5, type=float)
    parser.add_argument("--context_enlarge_ratio", default=0.5, type=float)
    parser.add_argument("--part_in_use_whole_prob", default=0.2, type=float)
    parser.add_argument("--part_in_prepend_object_class_prob", default=0.8, type=float)
    parser.add_argument("--mask2box_prob", default=0.0, type=float)
    parser.add_argument("--sliding_window_size", default=224, type=int)
    parser.add_argument("--sliding_window_stride", default=224, type=int)
    parser.add_argument("--backbone_output_stride", default=14, type=int)
    parser.add_argument("--backbone_output_channel", default=1408, type=int)
    parser.add_argument("--maskqformer_channel", default=768, type=int)
    parser.add_argument("--llm_channel", default=4096, type=int)
    
    # training args
    parser.add_argument(
        "--run_name",
        type=str,
        default="osm",
        help="used to name saving directory",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--batch_size_coco", type=int, default=0)
    parser.add_argument("--batch_size_lvis", type=int, default=0)
    parser.add_argument("--batch_size_v3det", type=int, default=0)
    parser.add_argument("--batch_size_a847", type=int, default=0)
    parser.add_argument("--batch_size_pc459", type=int, default=0)
    parser.add_argument("--batch_size_partimagenet", type=int, default=0)
    parser.add_argument("--batch_size_pascal_part", type=int, default=0)
    parser.add_argument("--batch_size_ade20k", type=int, default=0)
    parser.add_argument("--batch_size_cityscapes", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_coco", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_lvis", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_v3det", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_a847", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_pc459", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_partimagenet", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_pascal_part", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_ade20k", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_cityscapes", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples (train_num_samples_coco), not a pass through the entire dataset",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )

    # data args
    parser.add_argument(
        "--coco_shards",
        type=str,
        help="path to coco shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--lvis_shards",
        type=str,
        help="path to lvis shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--v3det_shards",
        type=str,
        help="path to v3det shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--a847_shards",
        type=str,
        help="path to a847 shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--pc459_shards",
        type=str,
        help="path to pc459 shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--partimagenet_shards",
        type=str,
        help="path to partimagenet shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--pascal_part_shards",
        type=str,
        help="path to pascal part shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--ade20k_shards",
        type=str,
        help="path to ade20k shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument(
        "--cityscapes_shards",
        type=str,
        help="path to cityscapes shards, this should be a glob pattern such as /path/to/shards/shard-{0000..0999}.tar",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_coco", type=int, default=10000)
    parser.add_argument("--train_num_samples_lvis", type=int, default=10000)
    parser.add_argument("--train_num_samples_v3det", type=int, default=10000)
    parser.add_argument("--train_num_samples_a847", type=int, default=10000)
    parser.add_argument("--train_num_samples_pc459", type=int, default=10000)
    parser.add_argument("--train_num_samples_partimagenet", type=int, default=10000)
    parser.add_argument("--train_num_samples_pascal_part", type=int, default=10000)
    parser.add_argument("--train_num_samples_ade20k", type=int, default=10000)
    parser.add_argument("--train_num_samples_cityscapes", type=int, default=10000)
    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp_use_orig_params",
        default=False,
        action="store_true",
        help="Passed into the FSDP constructor. Enables param_groups and gradient masking for weight_decay. Does not work with OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"]
    )

    args = parser.parse_args()

    if args.fsdp and not args.fsdp_use_orig_params:
        print(
            "Warning: FSDP is running without fsdp_use_orig_params flag. "
            + "This is not recommended because it means we will use uniform weight decay"
            + " and train all embeddings, not just the newly added ones. "
            + "Note: OPT models are not compatible with fsdp_use_orig_params flag."
        )

    if args.fsdp and args.fsdp_sharding_strategy == "hybrid":
        print(
            "Warning: As of torch=2.0.1, the FSDP logic for optim_state_dict() is broken for hybrid sharding."
            + "To make this method work, we need to modify torch.distributed.fsdp._optim_utils.py"
            + "Copy and paste the code from the _optim_utils.py in this repo into the torch file."
            + "The main issue was the missing group kwarg on line 1596 in _all_gather_optim_state."
        )

    # Set up distributed training
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize model
    model, processor = create_model_and_transforms(
        base_model_name=args.base_model_name,
        input_size=args.input_size,
        sliding_window_size=args.sliding_window_size,
        sliding_window_stride=args.sliding_window_stride,
        backbone_output_stride=args.backbone_output_stride,
        backbone_output_channel=args.backbone_output_channel,
        maskqformer_channel=args.maskqformer_channel,
        llm_channel=args.llm_channel)

    random_seed(args.seed, args.rank)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")
    # Load model checkpoint on CPU
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        # if args do not specify a checkpoint to resume from, check if checkpoints exist for this run
        # and automatically resume from the latest checkpoint
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        resume_from_epoch = checkpoint["epoch"] + 1

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )

        # init MixedPrecision
        if args.precision != "fp32":
            cast_dtype = get_mp_policy_dtype(args.precision)
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=cast_dtype,  # gradient communication
                buffer_dtype=cast_dtype,
            )
        else:
            mp_policy = None

        # init process groups
        if args.fsdp_sharding_strategy == "hybrid":
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
                _get_default_group()
            )
            args.my_group = intra_node_group  # for optimizer saving
            process_group = (intra_node_group, inter_node_group)  # for FSDP init
        else:
            args.my_group = None  # for optimizer saving
            process_group = None  # for FSDP init

        # init FSDP
        wrapper_kwargs = dict(
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=args.fsdp_use_orig_params,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        model.wrap_fsdp(wrapper_kwargs, device_id)
        ddp_model = model

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )
    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, FSDP)
            and not isinstance(m, CheckpointWrapper),
        )

    # Initialize optimizer
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        # apply weight decay only to params in the xattn layers
        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                # weight decay for all trainable params
                params_with_wd.append(p)
            return [
                {"params": params_with_wd, "weight_decay": args.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize), lr=args.learning_rate
        )
    else:
        # unclear if we should be using no weight decay or small weight decay for all parameters
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None:
        osd = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            osd = FSDP.optim_state_dict_to_load(osd, ddp_model, optimizer)
        optimizer.load_state_dict(osd)

    # Initialize data loaders
    coco_dataset = None
    lvis_dataset = None
    v3det_dataset = None
    a847_dataset = None
    pc459_dataset = None
    partimagenet_dataset = None
    pascal_part_dataset = None
    ade20k_dataset = None
    cityscapes_dataset = None
    train_num_samples = None
    train_batch_size = None
    if args.batch_size_coco > 0:
        coco_dataset = get_data(args, processor, "coco")
        train_num_samples = args.train_num_samples_coco
        train_batch_size = args.batch_size_coco
    if args.batch_size_lvis > 0:
        lvis_dataset = get_data(args, processor, "lvis")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_lvis
            train_batch_size = args.batch_size_lvis
    if args.batch_size_v3det > 0:
        v3det_dataset = get_data(args, processor, "v3det")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_v3det
            train_batch_size = args.batch_size_v3det
    if args.batch_size_a847 > 0:
        a847_dataset = get_data(args, processor, "a847")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_a847
            train_batch_size = args.batch_size_a847
    if args.batch_size_pc459 > 0:
        pc459_dataset = get_data(args, processor, "pc459")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_pc459
            train_batch_size = args.batch_size_pc459
    if args.batch_size_partimagenet > 0:
        partimagenet_dataset = get_data(args, processor, "partimagenet")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_partimagenet
            train_batch_size = args.batch_size_partimagenet
    if args.batch_size_pascal_part > 0:
        pascal_part_dataset = get_data(args, processor, "pascal_part")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_pascal_part
            train_batch_size = args.batch_size_pascal_part
    if args.batch_size_ade20k > 0:
        ade20k_dataset = get_data(args, processor, "ade20k")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_ade20k
            train_batch_size = args.batch_size_ade20k
    if args.batch_size_cityscapes > 0:
        cityscapes_dataset = get_data(args, processor, "cityscapes")
        if train_num_samples is None:
            train_num_samples = args.train_num_samples_cityscapes
            train_batch_size = args.batch_size_cityscapes

    total_training_steps = (
        (train_num_samples) // (train_batch_size * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Start training!
    ddp_model.train()

    coco_loader = None
    lvis_loader = None
    v3det_loader = None
    a847_loader = None
    pc459_loader = None
    partimagenet_loader = None
    pascal_part_loader = None
    ade20k_loader = None
    cityscapes_loader = None

    for epoch in range(resume_from_epoch, args.num_epochs):
        if coco_dataset is not None:
            coco_dataset.set_epoch(epoch)
            coco_loader = coco_dataset.dataloader
        if lvis_dataset is not None:
            lvis_dataset.set_epoch(epoch)
            lvis_loader = lvis_dataset.dataloader
        if v3det_dataset is not None:
            v3det_dataset.set_epoch(epoch)
            v3det_loader = v3det_dataset.dataloader
        if a847_dataset is not None:
            a847_dataset.set_epoch(epoch)
            a847_loader = a847_dataset.dataloader
        if pc459_dataset is not None:
            pc459_dataset.set_epoch(epoch)
            pc459_loader = pc459_dataset.dataloader
        if partimagenet_dataset is not None:
            partimagenet_dataset.set_epoch(epoch)
            partimagenet_loader = partimagenet_dataset.dataloader
        if pascal_part_dataset is not None:
            pascal_part_dataset.set_epoch(epoch)
            pascal_part_loader = pascal_part_dataset.dataloader
        if ade20k_dataset is not None:
            ade20k_dataset.set_epoch(epoch)
            ade20k_loader = ade20k_dataset.dataloader
        if cityscapes_dataset is not None:
            cityscapes_dataset.set_epoch(epoch)
            cityscapes_loader = cityscapes_dataset.dataloader

        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            processor=processor,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            coco_loader=coco_loader,
            lvis_loader=lvis_loader,
            v3det_loader=v3det_loader,
            a847_loader=a847_loader,
            pc459_loader=pc459_loader,
            partimagenet_loader=partimagenet_loader,
            pascal_part_loader=pascal_part_loader,
            ade20k_loader=ade20k_loader,
            cityscapes_loader=cityscapes_loader,
            device_id=device_id
        )
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)

    # save final checkpoint
    save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)


if __name__ == "__main__":
    main()
