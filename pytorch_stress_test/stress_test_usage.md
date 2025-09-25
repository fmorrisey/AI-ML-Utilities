# xx_pytorch_stress.py — Usage Guide

Purpose
- Lightweight PyTorch script to stress-test a GPU (e.g., RTX A4000) with synthetic AI/ML-style workloads:
  - matmul, batched_matmul, conv2d, mlp.

Prerequisites
- Python with a CUDA-enabled PyTorch build.
- Sufficient GPU memory for the chosen workload.
- Run on the machine with your RTX A4000 and verify with `nvidia-smi`.

Quick examples
- Large single matmul:
  ```bash
  python xx_stress.py --workload matmul --size 16000 --iterations 100 --warmup 5
  ```
- Batched matmul:
  ```bash
  python xx_stress.py --workload batched_matmul --size 4096 --batch 8 --iterations 200
  ```
- Conv2D-style workload:
  ```bash
  python xx_stress.py --workload conv2d --size 2048 --batch 4 --channels 64 --iterations 100
  ```
- MLP (simulates FFN layers, add --with_grad to simulate training):
  ```bash
  python xx_stress.py --workload mlp --size 16384 --batch 2 --iterations 50 --with_grad
  ```

Important flags
- --device: device (default: cuda). Use `--device cpu` for CPU runs.
- --workload: one of `matmul`, `batched_matmul`, `conv2d`, `mlp`.
- --size: matrix dim N or spatial size (default: 16000).
- --batch: batch size for batched workloads (default: 1).
- --channels: channels for conv / mlp (default: 3).
- --iterations: timed iterations (default: 1000).
- --warmup: warmup iterations (default: 10).
- --dtype: `float32` or `float16`.
- --amp: enable mixed precision autocast with float16.
- --with_grad: perform backward() to simulate training (higher memory).
- --seed: RNG seed.

Tips for RTX A4000
- Large sizes (e.g., 16000x16000) use lots of VRAM. Reduce `--size` or `--batch` if you hit OOM.
- Use `--dtype float16 --amp` to lower memory use and stress mixed-precision code paths.
- `--with_grad` increases memory significantly — use smaller sizes if enabled.
- Monitor GPU with `nvidia-smi` during runs.

Interpreting output
- The script prints average iteration time, iterations/sec, estimated GFLOPs, and memory usage.
- GFLOPs is an approximate metric based on operation counts — useful for comparisons across runs.

Avoiding / recovering from OOM
- Reduce `--size` and/or `--batch`.
- Switch to `--dtype float16` and add `--amp`.
- If OOM occurs, the script clears cache; you may need to restart the Python process to fully recover.

Suggested experiments
- Compute-bound vs memory-bound: increase size for memory pressure, increase batch for compute intensity.
- Mixed precision comparison: run same workload with `float32` and `float16 --amp`.
- Training vs inference: compare runs with and without `--with_grad`.

Troubleshooting
- "CUDA not available": ensure proper CUDA drivers and a CUDA-enabled PyTorch build.
- Repeated OOM: lower sizes/batch or use float16.
- Slow runs: confirm the script runs on GPU (device printed at startup).

Start small, verify stability, then scale up to find the stress point for your RTX A4000.
