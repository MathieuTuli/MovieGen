from dataclasses import asdict
from pathlib import Path

import argparse
import random
import signal
import time
import sys
import os


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.utils.data import DataLoader
from PIL import Image

# import torch.distributed as dist
import torch.distributed as dist
import numpy as np
import torchvision
import torch
import cv2

from metrics import collect_metrics
from tae import TAE, TAEConfig
from util import (
    dump_dict_to_yaml, asspath, mkpath, print0, cleanup, signal_handler)


class Dataset:
    def __init__(self, root: Path, T: int, image_only: bool,
                 size: int = 64, train: bool = True):
        self.train = train
        self.image_only = image_only
        self.T = T
        # Add shared state for batch-level decision
        self.use_image_mode = None
        self.files = list()
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.gif'):
            self.files.extend(root.glob(ext))
        self.files = sorted(self.files)  # Sort for consistency
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def load_video(self, fname):
        vidcap = cv2.VideoCapture(fname)
        ret, x = vidcap.read()
        x = Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        x = self.transforms(x)[None, :]
        while ret:
            ret, frame = vidcap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                x = torch.cat([x, self.transforms(frame)[None, :]])
        return x

    def __getitem__(self, index):
        # Make batch-level decision if not already made
        if self.use_image_mode is None:
            self.use_image_mode = self.image_only or (
                random.random() < 0.25 and self.train)

        video = self.load_video(self.files[index])
        vlen = video.shape[0]

        if self.use_image_mode:
            id = random.randint(0, vlen - 1) if self.train else 0
            frame = video[id]
            # NOTE: using 8 magic number for single frame
            zero_pad = torch.zeros(8 - 1, *frame.shape, dtype=frame.dtype)
            x = torch.cat([frame[None, :], zero_pad], dim=0)
        else:
            id = 0
            if self.train:
                id = random.randint(0, max(vlen - self.T - 1, 0))
            x = video[id:id + self.T]
        return x


"""
-------------------------------------------------------------------------------
args
-------------------------------------------------------------------------------
"""

parser = argparse.ArgumentParser()
# io
# yes you can use parser types like this
parser.add_argument("--output-dir", type=mkpath, default="")
parser.add_argument("--train-data-dir", type=asspath,
                    default="dev/data/train-overfit")
parser.add_argument("--val-data-dir", type=asspath,
                    default="dev/data/val-overfit")

# checkpointing
parser.add_argument("--ckpt", type=asspath, required=False)
parser.add_argument("--ckpt-from-ldm", type=int, default=0, choices=[0, 1])
parser.add_argument("--resume", type=int, default=0, choices=[0, 1])
parser.add_argument("--ckpt-freq", type=int, default=-1)
parser.add_argument("--device", type=str, default="cuda")

# optimization
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--total-batch-size", type=int, default=None)
parser.add_argument("--max-frames", type=int, default=32)
parser.add_argument("--resolution", type=int, default=64)
parser.add_argument("--image-only", type=int, default=0)

parser.add_argument("--num-iterations", type=int, default=10)
parser.add_argument("--val-loss-every", type=int, default=0)
parser.add_argument("--val-max-steps", type=int, default=20)

parser.add_argument("--inference-only", default=0, type=int, choices=[0, 1])
parser.add_argument("--compute-magic-number", default=0,
                    type=int, choices=[0, 1])
parser.add_argument("--verbose-loss", default=0, type=int, choices=[0, 1])
parser.add_argument("--seed", default=420, type=int)

# memory management
parser.add_argument("--dtype", type=str, default="float32")
parser.add_argument("--compile", default=0, type=int)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parser.parse_args()
    # args error checking
    assert args.dtype in {"float32"}

    train_logfile, val_logfile = None, None
    if args.output_dir:
        train_logfile = args.output_dir / "train.log"
        val_logfile = args.output_dir / "val.log"
        if args.resume == 0:
            open(train_logfile, 'w').close()
            open(val_logfile, 'w').close()
    else:
        args.output_dir = mkpath("tmp")

    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        assert torch.cuda.is_available(), "We need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    config = TAEConfig()
    config_dict = {'TAEConfig': asdict(config), 'CLIArgs': vars(args)}
    dump_dict_to_yaml(args.output_dir / "config_dump.yaml", config_dict)
    model = TAE(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print0(f"Total Parameters: {total_params:,}")
    print0(f"Trainable Parameters: {trainable_params:,}")
    if args.ckpt:
        ignore_keys = list()
        # The checkpoint from LDM needs to ignore certain layers
        if args.ckpt_from_ldm == 1:
            ignore_keys = [
                "encoder.conv_out",
                "decoder.conv_in",
                "quant_conv",
                "post_quant_conv",
                "loss",
            ]

        # if args.resume == 0:
        #     ignore_keys.append("loss")

        model.from_pretrained(args.ckpt, ignore_keys=ignore_keys)
        print0(f"Loaded ckpt {args.ckpt}")

    model.train()
    if args.compile:
        model = torch.compile(model)
    model.to(device)

    trainset = Dataset(args.train_data_dir, T=args.max_frames,
                       image_only=args.image_only == 1,
                       size=args.resolution)
    valset = Dataset(args.val_data_dir, T=args.max_frames,
                     image_only=args.image_only == 1,
                     size=args.resolution, train=False)
    if args.val_max_steps < 0:
        args.val_max_steps = len(valset)
    train_sampler = DistributedSampler(trainset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(valset) if ddp else None
    batch_size = args.batch_size
    grad_accum_steps = args.total_batch_size // ddp_world_size
    print0(f"Total desired batch size: {args.total_batch_size}")
    print0(f"=> batch size per rank: {batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    def train_worker_init_fn(worker_id):
        # Reset the shared state for each new batch
        trainset.use_image_mode = None

    def val_worker_init_fn(worker_id):
        # Reset the shared state for each new batch
        trainset.use_image_mode = None

    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=4, sampler=train_sampler,
                              worker_init_fn=train_worker_init_fn)
    val_loader = DataLoader(valset, batch_size=batch_size,
                            shuffle=False, num_workers=4, sampler=val_sampler,
                            worker_init_fn=val_worker_init_fn)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank],
                    find_unused_parameters=True)
    unwrapped_model = model.module if ddp else model

    optimizer_ae, optimizer_disc = unwrapped_model.configure_optimizers(
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.5, 0.9))

    torch.cuda.reset_peak_memory_stats()
    timings = list()
    if args.inference_only:
        print0("Starting inference only.")
    else:
        print0("Starting training.")

    if args.compute_magic_number:
        all_latents = list()
        with torch.no_grad():
            for x in train_loader:
                x = x.to(device)
                latents = model.encode(x).sample()
                all_latents.append(latents.cpu())
        local_latents = torch.cat(all_latents)

        if ddp:
            # Get the maximum size across all processes
            local_size = torch.tensor([local_latents.shape[0]], device=device)
            all_sizes = [torch.zeros_like(local_size)
                         for _ in range(ddp_world_size)]
            torch.distributed.all_gather(all_sizes, local_size)
            max_size = max(size.item() for size in all_sizes)

            # Pad local_latents to max_size
            if local_latents.shape[0] < max_size:
                padding = torch.zeros(max_size - local_latents.shape[0],
                                      *local_latents.shape[1:],
                                      dtype=local_latents.dtype,
                                      device=local_latents.device)
                local_latents = torch.cat([local_latents, padding])

            # Gather all latents
            all_gathered = [torch.zeros_like(
                local_latents) for _ in range(ddp_world_size)]
            torch.distributed.all_gather(all_gathered, local_latents)

            # On master process, combine and compute stats
            if master_process:
                # Remove padding and concatenate
                all_gathered = [tensor[:size.item()]
                                for tensor, size in zip(all_gathered,
                                                        all_sizes)]
                all_latents_tensor = torch.cat(all_gathered)
                std = all_latents_tensor.std().item()
                normalizer = 1 / std
                print0(f"Magic number: {normalizer=}")
        else:
            # Single process case
            std = local_latents.std().item()
            normalizer = 1 / std
            print0(f"Magic number: {normalizer=}")
        cleanup()
        sys.exit(0)

    start_step = 0
    if args.resume == 1:
        start_step = torch.load(args.ckpt)["step"]
        train_iter = iter(train_loader)
        if train_sampler is not None:
            train_sampler.set_epoch(start_step % len(trainset))
    trainset_size = len(trainset) // ddp_world_size if ddp else len(trainset)
    for step in range(start_step, args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations - 1)

        # once in a while evaluate the validation dataset
        if ((args.val_loss_every > 0 and step % args.val_loss_every == 0) or last_step or args.inference_only) and (val_loader is not None) and master_process:  # noqa
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                metrics = {
                    "psnr_image": list(),
                    "ssim_image": list(),
                    "psnr_video": list(),
                    "ssim_video": list(),
                }
                val_iter = iter(val_loader)
                for vali in range(args.val_max_steps):
                    try:
                        x = next(val_iter)
                    except StopIteration:
                        break
                    x = x.to(device)
                    dec, _, loss, loss_dict = model(x, "val", 1, vali)
                    val_loss += loss.item()
                    metrics = collect_metrics(
                        dec.permute(0, 1, 3, 4, 2).cpu().numpy(),
                        x.permute(0, 1, 3, 4, 2).cpu().numpy(),
                        metrics)
                    for b in range(dec.shape[0]):
                        for t in range(dec.shape[1]):
                            fn = args.output_dir /\
                                f"val_dec__step_{step}_vali_{vali}__B_{b}__T_{t}.png"  # noqa
                            torchvision.transforms.ToPILImage()(
                                (dec[b, t] + 1) * 0.5).save(fn)
                            fn = args.output_dir /\
                                f"val_x__step_{step}_vali_{vali}__B_{b}__T_{t}.png"  # noqa
                            torchvision.transforms.ToPILImage()(
                                (x[b, t] + 1) * 0.5).save(fn)
                val_loss /= vali
            # log to console and to file
            print0(f"val loss {val_loss}")
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            x = " | ".join([f'{x}: {y:.4f}' for x, y in metrics.items()])
            print0(f" val metrics: {x}")
            if args.verbose_loss:
                x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict.items()])  # noqa
                print0(f"    val verbose loss: {x}")
            if val_logfile is not None:
                with open(val_logfile, "a") as f:
                    f.write(f"{step},")
                    f.write(f"val/loss,{val_loss},")
                    f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict.items()]))  # noqa
                    f.write("\n")
                    f.write(",".join([f'{x},{y:.4f}' for x, y in metrics.items()]))  # noqa
                    f.write("\n")

        if last_step or args.inference_only:
            break

        model.train()
        # --------------- VAE TRAINING SECTION BEGIN -----------------
        optimizer_ae.zero_grad(set_to_none=True)
        loss_ae_accum, loss_dict_ae_accum = 0., dict()
        for micro_step in range(grad_accum_steps):
            if micro_step % trainset_size == 0:
                train_iter = iter(train_loader)
                if train_sampler is not None:
                    train_sampler.set_epoch(step % len(trainset))

            # fetch a batch
            x = next(train_iter)
            x = x.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way is with model.no_sync(), but that is a
                # context manager that bloats code, so we just toggle this
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1)
            dec, post, loss_ae, loss_dict_ae = model(x, "train", 0, step)
            loss_ae = loss_ae / grad_accum_steps
            for k, v in loss_dict_ae.items():
                if k not in loss_dict_ae_accum:
                    loss_dict_ae_accum[k] = v / grad_accum_steps
                else:
                    loss_dict_ae_accum[k] += v / grad_accum_steps
            loss_ae_accum += loss_ae.detach()
            loss_ae.backward()
        if ddp:
            dist.all_reduce(loss_ae_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        # step the optimizer
        optimizer_ae.step()

        # --------------- DISC TRAINING SECTION BEGIN -----------------
        optimizer_disc.zero_grad(set_to_none=True)
        loss_disc_accum, loss_dict_disc_accum = 0., dict()
        for micro_step in range(grad_accum_steps):
            if micro_step % trainset_size == 0:
                train_iter = iter(train_loader)
                if train_sampler is not None:
                    train_sampler.set_epoch(step % len(trainset))
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1)
            # fetch a batch
            dec, post, loss_disc, loss_dict_disc = model(x, "train", 1, step)
            loss_disc = loss_disc / grad_accum_steps
            for k, v in loss_dict_disc.items():
                if k not in loss_dict_disc_accum:
                    loss_dict_disc_accum[k] = v / grad_accum_steps
                else:
                    loss_dict_disc_accum[k] += v / grad_accum_steps
            loss_disc_accum += loss_disc.detach()
            loss_disc.backward()
        if ddp:
            dist.all_reduce(loss_disc, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        # step the optimizer
        optimizer_disc.step()
        # --------------- TRAINING SECTION END -------------------

        torch.cuda.synchronize()
        t1 = time.time()
        print0(f"""step {step+1:4d}/{args.num_iterations} | \
               train ae loss {loss_ae_accum.item():.6f} | \
               train disc loss {loss_disc_accum.item():.6f} | \
               norm {norm:.4f} | \
               """)
        if args.verbose_loss:
            x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict_ae_accum.items()])  # noqa
            print0(f"    ae verbose loss: {x}")
            x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict_disc_accum.items()])  # noqa
            print0(f"    disc verbose loss: {x}")
        if master_process and train_logfile is not None:
            with open(train_logfile, "a") as f:
                f.write(f"{step},")
                f.write(f"train/loss_ae,{loss_ae_accum},")
                f.write(f"train/loss_disc,{loss_disc_accum},")
                f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict_ae_accum.items()]))  # noqa
                f.write(",")
                f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict_disc_accum.items()]))  # noqa
                f.write("\n")

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

        if master_process and step > 0 and args.ckpt_freq > 0 and step % args.ckpt_freq == 0:
            torch.save({"step": step, "state_dict": unwrapped_model.state_dict()},
                       args.output_dir / f"tae_{step}.ckpt")

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")  # NOQA

    if master_process:
        torch.save({"step": step, "state_dict": model.state_dict()},
                   args.output_dir / "tae_last.ckpt")

    cleanup()
    # -------------------------------------------------------------------------
