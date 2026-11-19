# Softmax Fusion Lab

Original, bounded C++/CUDA engineering workbench for bias + scale + prefix-masked row softmax.

**Status: CPU reference/tests and host benchmark support validated. CUDA source is experimental and has NOT been compiled or run. GPU correctness, sanitizer checks and performance measurements are PENDING. This project is not fully validated or complete.** The current execution environment has GCC 14.2 but neither `nvcc` nor `nvidia-smi`. No GPU speedup or timing numbers are published.

## Question and contract

Does fusing bias/scale into a row-softmax kernel avoid an intermediate tensor and second launch without compromising numerical correctness? This is a question for actual GPU execution, not a conclusion from CPU tests.

For row `r`, valid columns are `c < lengths[r]`. Compute `z=(x[r,c]+bias[c])*scale`, then softmax over that prefix. All other outputs are exactly zero. A fully masked row returns all zeros by explicit convention, not a normalized distribution.

Bounded API: 0–1024 rows, 1–4096 columns, finite input/bias magnitudes ≤80, finite scale in `(0,4]`. Invalid values are rejected even in masked cells. The GPU benchmark deliberately requires at least one row. No arbitrary strides/masks, backward pass, FP16, aliasing or training integration.

## CPU setup and demo

Only a C++17 compiler and standard library are required:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror softmax_cpu.cpp -o /tmp/softmax_cpu
/tmp/softmax_cpu test
/tmp/softmax_cpu demo
/tmp/softmax_cpu study
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror test_bench.cpp -o /tmp/test_bench
/tmp/test_bench
```

Observed demo:

```text
CPU prefix-softmax:
0.0776955783 0.348207414 0.574096978 0
0 0 0 0
```

The CPU candidate uses the established serial online max/normalizer recurrence and recomputes logits in a second pass. Its independent reference materializes long-double logits and exponentials with a conventional max-shifted sum. Neither is a CUDA simulator.

The fixed CPU study covers 21 widths × three original xorshift32 seeds × four input modes, five prefix lengths per matrix: 252 matrices / 1260 rows. Widths include 31/32/33, 255/256/257, 1023/1024/1025 and 4095/4096. Fully masked/singleton and repeated narrow-width lengths are intentional. Diagnostic cases are not a representative workload distribution.

Initial GCC 14.2 CPU measurements: float mantissa 24 bits, reference mantissa 64 bits; maximum absolute probability error `1.0926696584369847e-7`, maximum row-mass error `7.31821885510285011e-6`. No GPU or performance inference follows. Checks use absolute tolerance `1e-4 + 1e-3*abs(reference)` and row-mass tolerance `1e-5`, plus finite/nonnegative outputs and exact masked zeros. Long-double precision varies by platform.

## Experimental CUDA build and execution

The following is a runnable entry point intended for an **existing compatible, genuinely free CUDA environment**, but these commands have NOT been executed here. A CPU-only compile or source inspection does not validate CUDA. Do not purchase/rent compute or enable a credit-card trial for this project.

```bash
nvcc -std=c++17 -O2 softmax_cuda.cu -o gpu_bench
./gpu_bench 128 257 10 50
compute-sanitizer --tool memcheck ./gpu_bench 8 33 0 1
compute-sanitizer --tool racecheck ./gpu_bench 8 33 0 1
compute-sanitizer --tool synccheck ./gpu_bench 8 33 0 1
```

The CUDA toolkit, host compiler and target GPU must be mutually compatible. The command uses the compiler's default architecture; record compiler flags and select a supported target when required by the available hardware. No `--use_fast_math` flag is used. Missing compiler, unsupported device or failed checks are blockers, not successful validation. No GPU-enabled CI or unattended device access is assumed.

### Architecture and baseline

`softmax_cuda.cu` supplies two pipelines:

- **Fused:** one 256-thread block per row; bias/scale computed inside masked max, sum and output passes. Shared-memory binary-tree reductions and block-wide barriers. No materialized logits tensor.
- **Materialized baseline:** a separate bias/scale kernel writes a full intermediate tensor, followed by the same row-reduction structure reading those logits. Two launches. This is a controlled engineering baseline, not cuDNN/PyTorch or an optimized competitor.

Unlike the serial CPU candidate, CUDA uses ordinary max-then-sum reductions, not a parallel online-normalizer reduction. All threads reach every shared reduction barrier; the fully masked return is uniform per block. Columns beyond the valid prefix contribute no mass and are written as zero. The harness validates the host inputs before copying and launching. Kernels themselves assume validated pointers/shapes/domain and are not an independent public unchecked API.

To reuse the established CPU reference without moving or copying it, the single-translation-unit host test and CUDA driver include `softmax_cpu.cpp` with `FUSION_NO_MAIN`; that guard is the only change to the initial source. `bench_common.hpp` provides shared argument parsing, workload generation, output validation and timing summaries. This is intentionally a small educational build, not a general linkable library.

### Benchmark protocol

Arguments are `rows columns warmup repetitions`; defaults `128 257 10 50`. Bounds are rows 1–1024, columns 1–4096, warmup 0–100, repetitions 1–1000. Fixed seed 2026 generates the selected workload; row lengths cycle through zero, one, half and full width. With a single row the workload is fully masked, so that is not a useful nonempty-row throughput case.

Before reporting timings, both variants run all 252 original diagnostic matrices, an extreme-scale case and the selected workload against the long-double CPU reference. Each measured variant checks correctness before warmup and again after timing. Failure exits nonzero. Never interpret a partial log ending in failure as a successful benchmark suite.

Warmup launches are followed by device synchronization. Each repetition uses start/stop CUDA events on the default stream and waits for the stop event. Reported min/median/mean/max are **kernel-pipeline event intervals** in milliseconds. They include device idle/host enqueue gaps between event records and kernels; they are not isolated instruction latency. Allocation, transfers, CPU reference, output validation and event creation are outside timing. No end-to-end timings are provided. Variants run in fixed order, not randomized; cache, clocks, thermal state and other device users may confound comparisons.

Metadata includes device name/SM/global memory, driver/runtime version, NVCC/host compiler version, shape, block size, warmup, repetition count and numerical errors. Save the exact build command and full stdout/stderr with any future measurements. No CUDA compilation or device measurement has yet established that this implementation works or is faster.

## Tests, limitations and next gate

### Source-bound report capture (Python 3.12, standard library)

On actual compatible hardware, after building and reviewing the GPU executable:

```bash
python benchmark_report.py capture ./gpu_bench local-run.json --build-command 'nvcc -std=c++17 -O2 softmax_cuda.cu -o gpu_bench' --shape 128 257 10 50
python benchmark_report.py validate local-run.json
python -W error -m unittest -v test_report
```

Capture executes the selected binary directly, without a shell, with a 180-second timeout. It records source hashes for the kernel/CPU reference/shared harness, the executable digest, caller-recorded build command and complete stdout/stderr. It requires unchanged sources/binary during execution, exit code zero, empty stderr, exact timing scope, metadata, paired variant shapes/counts, positive ordered timing summaries, bounded numerical errors and the final success marker. Files are created exclusively, never overwritten; failed capture creates no report. Reports require at least two rows, one warmup and two repetitions, so sanitizer-only/zero-warmup runs are not accepted as useful benchmark records. Capture does not execute the supplied build-command string.

Validation reparses the raw log and checks the saved fields/digest against local source hashes. This is **internal consistency, not hardware attestation**, proof that the binary was built from those sources, or independent verification of device correctness. A forged process log can still be internally consistent. Preserve independent compiler/device-sanitizer evidence before making GPU claims. No actual GPU report is shipped here. All parser fixtures and mocked subprocess tests in `test_report.py` are explicitly **SYNTHETIC UNIT TESTS**, not measurements; they produce temporary files only.

Seven CPU test groups exercise analytic probabilities, extreme/non-dyadic values, mask conventions, invariances, invalid inputs, empty batches, immutability and the full numerical study. Host-harness tests exercise independently known timing statistics, bounded parsing, deterministic workloads, correctness failure injection and nine boundary widths. These tests do **not** execute kernel synchronization, device reductions or CUDA API paths.

```bash
g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer test_bench.cpp -o /tmp/test_bench_san
ASAN_OPTIONS=detect_leaks=0 /tmp/test_bench_san
```

LeakSanitizer is disabled in the current runtime because process inspection/ptrace is restricted. Address/undefined-behaviour checks can run; leak detection is not claimed. CI runs CPU and host-harness checks only, never a fake GPU pass.

Next acceptance gate: compile on actual compatible free hardware, run all numerical cases and device memcheck/racecheck/synccheck, then collect repeated measurements with metadata. Until then, the fifth project's GPU portion and full completion remain pending.

## Attribution and license

Original source and generated inputs, MIT licensed; see [LICENSE](LICENSE). No third-party implementation, assets, datasets or private research are incorporated. The online-normalizer mathematical idea is attributed to [Milakov and Gimelshein (2018)](https://arxiv.org/abs/1805.02867). CUDA synchronization/timing methodology follows [NVIDIA's official Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html). No paper performance claims are adopted as results here.
