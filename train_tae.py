from pathlib import Path

import argparse
import random
import signal
import time
import os


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

import torch.distributed as dist
import numpy as np
import torchvision
import torch
import cv2

from metrics import collect_metrics
from tae import TAE, TAEConfig


class DistributedDataLoader:
    def __init__(self,
                 root: Path,
                 B: int, T: int,
                 rank: int, world_size: int,
                 size: int = 64, train: bool = True):
        self.train = train
        self.B, self.T, self.rank, self.world_size = B, T, rank, world_size
        self.files = list()
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.gif'):
            self.files.extend(root.glob(ext))
        self.files = sorted(self.files)  # Sort for consistency
        assert len(self.files) >= world_size * B, \
            f"Have {len(self.files)}, need {world_size * B} files"

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])

        self.count = -1
        self.indices = [x for x in range(0, len(self.files))]
        if train:
            random.shuffle(self.indices)

    def reset(self):
        self.count = self.rank * self.B

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

    def next_batch(self):
        # you're always safe to read the B videos from indices
        x, mask = torch.empty(0), torch.ones([self.B, self.T])
        for i in range(self.B):
            video = self.load_video(self.files[self.indices[self.count]])
            vlen = video.shape[0]
            # 1:3 image/video ratio from paper
            if self.count % 4 == 0:
                id = random.randint(0, vlen - 1) if self.train else 0
                frame = video[id]
                zero_pad = torch.zeros(1, self.T - 1, *frame.shape,
                                       dtype=frame.dtype)
                frame = torch.cat([frame[None, None, :], zero_pad], dim=1)
                x = torch.cat([x, frame])
                mask[i, 1:] = 0
            else:
                id = 0
                if self.train:
                    id = random.randint(0, max(vlen - self.T - 1, 0))
                frames = video[id:id + self.T]
                if frames.shape[0] < self.T:
                    zero_pad = torch.zeros(self.T - frames.shape[0],
                                           *frames.shape[1:],
                                           dtype=frame.dtype)
                    frames = torch.cat([frames, zero_pad], dim=0)
                x = torch.cat([x, frames[None, :]])
                mask[i, x.shape[1]:] = 0
            self.count += 1
            if self.count >= len(self.files):
                self.count = self.rank * self.B
        self.count += self.world_size * self.B
        # this will lose some videos at the end but who cares
        if self.count + self.B >= len(self.files):
            self.count = self.rank * self.B

        return x, mask


"""
-------------------------------------------------------------------------------
helpers
-------------------------------------------------------------------------------
"""


def asspath(strarg):
    """helper to ensure arg path exists"""
    p = Path(strarg)
    if p.exists():
        return p
    else:
        raise NotADirectoryError(strarg)


def mkpath(strarg):
    """helper to mkdir arg path if it doesn't exist"""
    if not strarg:
        return ""
    p = Path(strarg)
    p.mkdir(exist_ok=True, parents=True)
    return p


def print0(*args, **kwargs):
    """modified print that only prints from the master process"""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def cleanup():
    """Cleanup function to destroy the process group"""
    if int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print0("\nCtrl+C caught. Cleaning up...")
    cleanup()
    exit(0)


"""
-------------------------------------------------------------------------------
args
-------------------------------------------------------------------------------
"""

parser = argparse.ArgumentParser()
# io
# yes you can use parser types like this
parser.add_argument("--output-dir", type=mkpath, default="")
parser.add_argument("--train-dir", type=asspath, default="dev/data/train-smol")
parser.add_argument("--val-dir", type=asspath, default="dev/data/val-smol")

# checkpointing
parser.add_argument("--ckpt", type=asspath, required=False)
parser.add_argument("--ckpt-from-ldm", type=int, default=0, choices=[0, 1])
parser.add_argument("--resume", type=int, default=0, choices=[0, 1])
parser.add_argument("--ckpt-freq", type=int, default=-1)
parser.add_argument("--device", type=str, default="cuda")

# optimization
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--max-frames", type=int, default=32)
parser.add_argument("--resolution", type=int, default=64)

parser.add_argument("--num-iterations", type=int, default=10)
parser.add_argument("--val-loss-every", type=int, default=0)
parser.add_argument("--val-max-steps", type=int, default=20)
parser.add_argument("--overfit-batch", default=1, type=int)

parser.add_argument("--inference-only", default=0, type=int, choices=[0, 1])
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
            ]

        # if args.resume == 0:
        #     ignore_keys.append("loss")

        model.from_pretrained(args.ckpt, ignore_keys=ignore_keys)

    model.train()
    if args.compile:
        model = torch.compile(model)
    model.to(device)

    batch_size = args.batch_size
    if ddp:
        batch_size = args.batch_size // ddp_world_size
    train_loader = DistributedDataLoader(
            args.train_dir, B=batch_size, T=args.max_frames,
            rank=ddp_rank, world_size=ddp_world_size,
            size=args.resolution)
    val_loader = DistributedDataLoader(
            args.val_dir, B=batch_size, T=args.max_frames,
            rank=ddp_rank, world_size=ddp_world_size,
            size=args.resolution, train=False)

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

    start_step = 0
    if args.resume == 1:
        start_step = torch.load(args.ckpt)["step"]
    for step in range(start_step, args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations - 1)

        # once in a while evaluate the validation dataset
        if ((args.val_loss_every > 0 and step % args.val_loss_every == 0) or last_step or args.inference_only) and (val_loader is not None) and master_process:  # noqa
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                val_loader.reset()
                metrics = {
                    "psnr_image": list(),
                    "ssim_image": list(),
                    "psnr_video": list(),
                    "ssim_video": list(),
                }
                for vali in range(args.val_max_steps):
                    x, mask = val_loader.next_batch()
                    x, mask = x.to(device), mask.to(device)
                    dec, _, loss, loss_dict = model(x, "val", 1, step)
                    val_loss += loss.item()
                    metrics = collect_metrics(
                        dec.permute(0, 1, 3, 4, 2).cpu().numpy(),
                        x.permute(0, 1, 3, 4, 2).cpu().numpy(),
                        metrics)
                    for b in range(dec.shape[0]):
                        for t in range(dec.shape[1]):
                            fn = args.output_dir /\
                                f"val_dec_{vali}_{b}_{t}.png"
                            torchvision.transforms.ToPILImage()(
                                (dec[b, t] + 1) * 0.5).save(fn)
                            fn = args.output_dir / f"val_x_{vali}_{b}_{t}.png"
                            torchvision.transforms.ToPILImage()(
                                (x[b, t] + 1) * 0.5).save(fn)
                val_loss /= args.val_max_steps
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

        # fetch a batch
        x, mask = train_loader.next_batch()
        x, mask = x.to(device), mask.to(device)
        dec, post, loss_ae, loss_dict_ae = model(x, "train", 0, step)
        loss_ae.backward()
        # if ddp:
        #     dist.all_reduce(loss_ae, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        # step the optimizer
        optimizer_ae.step()

        # --------------- DISC TRAINING SECTION BEGIN -----------------
        optimizer_disc.zero_grad(set_to_none=True)

        # fetch a batch
        # REVISIT: should I be getting the next batch here?
        # x, mask = train_loader.next_batch()
        # x, mask = x.to(device), mask.to(device)

        dec, post, loss_disc, loss_dict_disc = model(x, "train", 1, step)
        loss_disc.backward()
        # # if ddp:
        # #     dist.all_reduce(loss_disc, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        # step the optimizer
        optimizer_disc.step()
        # --------------- TRAINING SECTION END -------------------

        torch.cuda.synchronize()
        t1 = time.time()
        print0(f"""step {step+1:4d}/{args.num_iterations} | \
               train ae loss {loss_ae.item():.6f} | \
               train disc loss {loss_disc.item():.6f} | \
               norm {norm:.4f} | \
               """)
        if args.verbose_loss:
            x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict_ae.items()])  # noqa
            print0(f"    ae verbose loss: {x}")
            x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict_disc.items()])  # noqa
            print0(f"    disc verbose loss: {x}")
        if master_process and train_logfile is not None:
            with open(train_logfile, "a") as f:
                f.write(f"{step},")
                f.write(f"train/toss_ae,{loss_ae.item()},")
                f.write(f"train/toss_disc,{loss_disc.item()}")
                f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict_ae.items()]))  # noqa
                f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict_disc.items()]))  # noqa
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
