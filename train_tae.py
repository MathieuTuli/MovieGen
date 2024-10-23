from pathlib import Path

import argparse
import time
import math
import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import numpy as np
import torch

from tae import TAE, TAEConfig


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


"""
-------------------------------------------------------------------------------
args
-------------------------------------------------------------------------------
"""

parser = argparse.ArgumentParser()
# io
# yes you can use parser types like this
parser.add_argument("--ckpt", type=asspath, required=False)
parser.add_argument("--output-dir", type=mkpath, default="")
parser.add_argument("--ul2-ckpt", type=asspath, required=False)
parser.add_argument("--byt5-ckpt", type=asspath, required=False)
parser.add_argument("--metaclip-ckpt", type=asspath, required=False)
parser.add_argument("--tae-ckpt", type=asspath, required=False)
# optimization
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--total-batch-size", type=int, default=1)
parser.add_argument("--sequence-length", type=int, default=64)

parser.add_argument("--num-iterations", type=int, default=10)
parser.add_argument("--val-loss-every", type=int, default=0)
parser.add_argument("--val-max-steps", type=int, default=20)
parser.add_argument("--overfit-batch", default=False, action="store_true")
# memory management
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--compile", default=False, action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    # args error checking
    assert args.dtype in {"float32", "float16", "bfloat16"}

    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / \
            (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    logfile = None
    if args.output_dir:
        logfile = args.output_dir / "main.log"

    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{ddp_local_rank}')
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = 0  # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = torch.device("cuda")

    B, T = args.batch_size, args.sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16,
               'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    torch.manual_seed(420)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(420)

    config = MovieGenConfig()
    model = MovieGen(config)
    if args.compile:
        model = torch.compile(model)

    train_loader = ...
    val_loader = ...

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    optimizer = model.configure_optimizers(lr=args.lr,
                                           weight_decay=args.weight_decay,
                                           betas=(0.9, 0.95),
                                           device_type=device)

    torch.cuda.reset_peak_memory_stats()
    timings = list()
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations - 1)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0
                and (step % args.val_loss_every == 0 or last_step)) \
                and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d vl:%f\n" % (step, val_loss))

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.overfit_single_batch:
            train_loader.reset()

        lossf = 0.
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / grad_accum_steps
                lossf += loss.detach()  # keep track of the mean loss
            loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------

        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_second = \
            grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(f"""step {step+1:4d}/{args.num_iterations} | \
               train loss {lossf:.6f} | \
               norm {norm:.4f} | \
               lr {lr:.2e} | \
               ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s) \
               """)
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d tl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")  # NOQA

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
