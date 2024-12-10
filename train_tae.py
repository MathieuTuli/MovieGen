from pathlib import Path

import argparse
import random
import time
import os

from PIL import Image

import numpy as np
import torchvision
import torch
import cv2

from tae import TAE, TAEConfig


class DataLoader:
    def __init__(self, fname, size: int = 64):
        vidcap = cv2.VideoCapture(fname)
        ret, x = vidcap.read()
        x = Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            torchvision.transforms.Normalize([0.5], [0.5]),
            ])
        x = transforms(x)[None, :]
        while ret:
            ret, frame = vidcap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                x = torch.cat([x, transforms(frame)[None, :]])
        self.x = x
        self.count = -1

    def next_batch(self):
        self.count += 1
        if self.count % 4 == 0:
            id = random.randint(0, self.x.shape[0] - 1)
            return self.x[id][None, None, :]
        else:
            id = random.randint(0, self.x.shape[0] - 1)
            return self.x[None, id:id+32]


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
parser.add_argument("--ckpt-from-ldm", type=int, default=1, choices=[0, 1])
# parser.add_argument("--tae-ckpt", type=asspath, required=False)
# optimization
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=1)

parser.add_argument("--num-iterations", type=int, default=10)
parser.add_argument("--val-loss-every", type=int, default=0)
parser.add_argument("--val-max-steps", type=int, default=20)
parser.add_argument("--overfit-batch", default=1, type=int)
parser.add_argument("--inference-only", default=0, type=int, choices=[0, 1])
parser.add_argument("--verbose-loss", default=0, type=int, choices=[0, 1])
# memory management
parser.add_argument("--dtype", type=str, default="float32")
parser.add_argument("--compile", default=0, type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    # args error checking
    assert args.dtype in {"float32", "float16", "bfloat16"}

    logfile = None
    if args.output_dir:
        logfile = args.output_dir / "main.log"

    device = torch.device("cuda")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16,
               'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    torch.manual_seed(420)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(420)

    config = TAEConfig()
    model = TAE(config)
    if args.ckpt:
        ignore_keys = list()
        if args.ckpt_from_ldm:
            ignore_keys = [
                    "encoder.conv_out",
                    "decoder.conv_in",
                    "quant_conv",
                    "post_quant_conv",
                    ]

        model.from_pretrained(args.ckpt, ignore_keys=ignore_keys)

    if args.compile:
        model = torch.compile(model)
    model.to(device)

    train_loader = DataLoader("rickroll.gif")
    val_loader = DataLoader("rickroll.gif")

    optimizer_ae, optimizer_disc = model.configure_optimizers(
            lr=args.lr, weight_decay=args.weight_decay,
            betas=(0.5, 0.9))

    torch.cuda.reset_peak_memory_stats()
    timings = list()
    if args.inference_only:
        print("Starting inference only.")
    else:
        print("Starting training.")
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations - 1)

        # once in a while evaluate the validation dataset
        if ((args.val_loss_every > 0 and step % args.val_loss_every == 0) or last_step or args.inference_only) and (val_loader is not None):  # noqa
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for vali in range(args.val_max_steps):
                    x = val_loader.next_batch()
                    x = x.to(device)
                    dec, _, loss, loss_dict = model(x, "val", 1, step)
                    val_loss += loss.item()
                    for b in range(dec.shape[0]):
                        for t in range(dec.shape[1]):
                            fn = args.output_dir / f"val_dec_{vali}_{b}_{t}.png"
                            torchvision.transforms.ToPILImage()(
                                    (dec[b, t] + 1) * 0.5).save(fn)
                            fn = args.output_dir / f"val_x_{vali}_{b}_{t}.png"
                            torchvision.transforms.ToPILImage()(
                                    (x[b, t] + 1) * 0.5).save(fn)
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if args.verbose_loss:
                x = " | ".join([f'{x}: {y.item():.4f}' for x, y in loss_dict.items()])  # noqa
                print0(f"    val verbose loss: {x}")
            if logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d val_loss:%f\n" % (step, val_loss))
                    f.write(f"{step},")
                    f.write(f"val_loss,{val_loss},")
                    f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict.items()]))  # noqa
                    f.write("\n")

        if last_step or args.inference_only:
            break

        model.train()
        # --------------- VAE TRAINING SECTION BEGIN -----------------
        optimizer_ae.zero_grad(set_to_none=True)

        # fetch a batch
        x = train_loader.next_batch()
        x = x.to(device)
        with ctx:
            dec, post, loss_ae, loss_dict_ae = model(x, "train", 0, step)
        loss_ae.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip)
        # step the optimizer
        optimizer_ae.step()
        # --------------- DISC TRAINING SECTION BEGIN -----------------
        optimizer_disc.zero_grad(set_to_none=True)

        # fetch a batch
        x = train_loader.next_batch()
        x = x.to(device)

        with ctx:
            dec, post, loss_disc, loss_dict_disc = model(x, "train", 1, step)
        loss_disc.backward()
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
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"{step},")
                f.write(f"loss_ae,{loss_ae.item()},")
                f.write(f"loss_disc,{loss_disc.item()}")
                f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict_ae.items()]))  # noqa
                f.write(",".join([f'{x},{y.item():.4f}' for x, y in loss_dict_disc.items()]))  # noqa
                f.write("\n")

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")  # NOQA

    torch.save({"state_dict": model.state_dict()},
               args.output_dir / "tae.ckpt")

    # -------------------------------------------------------------------------
