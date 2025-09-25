import argparse
import time
import math
import torch
import torch.nn as nn
from contextlib import contextmanager

def parse_args():
    p = argparse.ArgumentParser(description="PyTorch stress tester for GPU (matrix ops / conv / mlp).")
    p.add_argument("--device", default="cuda", help="Device to run on (default: cuda).")
    p.add_argument("--workload", choices=["matmul", "batched_matmul", "conv2d", "mlp"], default="matmul",
                   help="Type of synthetic workload to run.")
    p.add_argument("--size", type=int, default=16000,
                   help="Matrix size N for NxN matmul or spatial size for conv (default: 16000).")
    p.add_argument("--batch", type=int, default=1,
                   help="Batch size for batched matmul / mlp / conv examples (default: 1).")
    p.add_argument("--channels", type=int, default=3,
                   help="Channels for conv / mlp (default: 3).")
    p.add_argument("--iterations", type=int, default=1000,
                   help="Number of timed iterations (default: 1000).")
    p.add_argument("--warmup", type=int, default=10,
                   help="Number of warmup iterations (default: 10).")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="Tensor dtype (default: float32). Use float16 to stress memory/bandwidth and amp.")
    p.add_argument("--amp", action="store_true", help="Use torch.cuda.amp.autocast for mixed precision.")
    p.add_argument("--with_grad", action="store_true", help="Enable autograd (backward) to simulate training.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return p.parse_args()

@contextmanager
def maybe_autocast(enabled, dtype):
    if enabled and dtype == torch.float16:
        with torch.cuda.amp.autocast():
            yield
    else:
        yield

def human(n):
    # simple human readable formatting for bytes
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU. This will not stress the RTX A4000.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Print configuration
    print("=== Stress test config ===")
    print(f"device={device}, workload={args.workload}, size={args.size}, batch={args.batch}, channels={args.channels}")
    print(f"dtype={dtype}, amp={args.amp}, with_grad={args.with_grad}, warmup={args.warmup}, iters={args.iterations}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        # Build workload
        if args.workload == "matmul":
            N = args.size
            # single large matrix multiply: A @ B where both are NxN
            A = torch.randn(N, N, device=device, dtype=dtype)
            B = torch.randn(N, N, device=device, dtype=dtype)
            flops_per_iter = 2.0 * (N ** 3)  # approx FLOPs for matmul NxN x NxN
            def step():
                return A @ B

        elif args.workload == "batched_matmul":
            Bsz = args.batch
            N = args.size
            # batched: (B,N,N) x (B,N,N) -> (B,N,N)
            A = torch.randn(Bsz, N, N, device=device, dtype=dtype)
            B = torch.randn(Bsz, N, N, device=device, dtype=dtype)
            flops_per_iter = 2.0 * Bsz * (N ** 3)
            def step():
                return torch.matmul(A, B)

        elif args.workload == "conv2d":
            # simulate a conv2d heavy op: large spatial dims with a kernel
            Bsz = args.batch
            C = args.channels
            H = W = args.size  # large spatial size
            out_channels = C
            kernel = 3
            x = torch.randn(Bsz, C, H, W, device=device, dtype=dtype)
            conv = nn.Conv2d(C, out_channels, kernel_size=kernel, padding=1).to(device).to(dtype)
            # rough FLOPs: 2 * Cout * Cin * H * W * kH * kW
            flops_per_iter = 2.0 * out_channels * C * H * W * kernel * kernel
            def step():
                return conv(x)

        elif args.workload == "mlp":
            Bsz = args.batch
            D = args.size
            hidden = max(1024, D // 4)
            # simple 2-layer MLP with GELU to simulate transformer feed-forward
            x = torch.randn(Bsz, D, device=device, dtype=dtype)
            linear1 = nn.Linear(D, hidden).to(device).to(dtype)
            linear2 = nn.Linear(hidden, D).to(device).to(dtype)
            gelu = nn.GELU()
            # approximate FLOPs: 2*(D*hidden + hidden*D) per sample
            flops_per_iter = 2.0 * Bsz * (D * hidden + hidden * D)
            def step():
                return linear2(gelu(linear1(x)))

        else:
            raise ValueError("unknown workload")

        # Warmup
        print("Running warmup...")
        for _ in range(args.warmup):
            with maybe_autocast(args.amp, dtype):
                out = step()
                if args.with_grad:
                    # small fake loss/backward to simulate training
                    if out.dtype.is_floating_point:
                        loss = out.sum()
                        loss.backward()
                        # zero grads to avoid growth
                        for p in []:  # model-free ops have no params; linear/conv do
                            if hasattr(p, "grad") and p.grad is not None:
                                p.grad.detach_()
                                p.grad.zero_()
                # ensure completion
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed loop
        print("Starting timed iterations...")
        t0 = time.time()
        total_flops = 0.0
        iter_times = []
        for i in range(args.iterations):
            it0 = time.time()
            try:
                with maybe_autocast(args.amp, dtype):
                    out = step()
                    if args.with_grad:
                        if out.dtype.is_floating_point:
                            loss = out.sum()
                            loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("OutOfMemory encountered. Consider reducing --size or --batch. Clearing cache and exiting.")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise
                else:
                    raise
            it1 = time.time()
            dt = it1 - it0
            iter_times.append(dt)
            total_flops += flops_per_iter
            # Simple per-iteration print every 10%
            if (i + 1) % max(1, args.iterations // 10) == 0:
                avg = sum(iter_times) / len(iter_times)
                iters_per_sec = 1.0 / avg if avg > 0 else float("inf")
                gflops = (total_flops / len(iter_times)) / 1e9
                mem_alloc = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                mem_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                print(f"iter {i+1}/{args.iterations} avg_iter={avg:.4f}s iters/s={iters_per_sec:.2f} est_GFLOPs={gflops:.2f} mem={human(mem_alloc)} peak={human(mem_peak)}")

        t1 = time.time()
        total_time = t1 - t0
        avg_iter = total_time / args.iterations if args.iterations else 0.0
        achieved_gflops = (total_flops / total_time) / 1e9 if total_time > 0 else 0.0

        print("\n=== Summary ===")
        print(f"Total time: {total_time:.2f}s, avg per iter: {avg_iter:.4f}s")
        print(f"Estimated achieved GFLOPs: {achieved_gflops:.2f} GFLOP/s")
        if device.type == "cuda":
            print(f"Final mem allocated: {human(torch.cuda.memory_allocated(device))}, peak: {human(torch.cuda.max_memory_allocated(device))}")

    except RuntimeError as e:
        print("RuntimeError:", e)
    finally:
        # cleanup to free GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
