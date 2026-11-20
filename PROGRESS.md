# Task-owned continuity

Fifth November project: https://github.com/BLANK-2340/softmax-fusion-lab. Public creation verified November 19, 2026, owner BLANK-2340 (105676435), repository 1463846170. First CPU foundation: ed82c7a234dfca5197935dc8b094fbe206e5852b.

## November 19, second session

Added experimental fused CUDA and materialized-baseline kernels, a correctness-gated device-event benchmark driver, tested host benchmark support, CPU CI, README architecture/protocol/results/limits and separate MIT license. The initial CPU source is preserved except a two-line `FUSION_NO_MAIN` guard so single-translation-unit drivers can reuse it directly.

CPU and host tests are the validation scope for this change. CUDA compilation, device correctness, memcheck/racecheck/synccheck and measured timings remain **PENDING** because the runtime has neither nvcc nor nvidia-smi. No simulated/stub build is represented as CUDA compilation. No GPU timings, speedup or fifth-project completion claimed. The earlier source comment saying no CUDA source/docs/CI is historical, superseded by this record.

Before publication: strict CPU and host builds/tests, sanitizer reruns with leak detection disabled, final diff and clean committed archive must pass. After publication: verify remote head/tree and CPU CI. Remote acceptance details are recorded in the automation continuity state after verification.

Next: use actual compatible existing free CUDA hardware when available, compile the exact experimental source, resolve any compiler/runtime failures, execute all device correctness and sanitizer checks and only then measure. While hardware remains unavailable, meaningful next host work is a machine-readable, source-bound benchmark-report validator that rejects failed/incomplete logs and verifies timing scope/counts/metadata; do not fabricate hardware evidence or spend sessions on documentation-only churn. Four other bounded projects and the three required public demos were already completed; all-five full completion remains outstanding.

## November 19, third session

Added dependency-free Python source-bound process-log capture and report validation, with seven unittest groups and CPU CI coverage. The CUDA kernel, C++ CPU reference and host harness are unchanged. Capture checks requested versus reported counts, exit/stderr/final-success, paired variants, finite consistent timing/error statistics, metadata/scope, source/binary stability and exclusive output creation. Validation checks exact local source hashes, raw-log digest and parsed-field correspondence. Provenance checks are explicitly not hardware attestation or proof of a binary/source build relationship.

Tests use clearly labeled synthetic log fixtures and mocked subprocesses only. No GPU execution or real report generation occurred. CUDA compilation/device correctness/sanitizers/timings remain pending; fifth-project completion remains unverified. Publication requires tests, clean-archive reruns, final diff and remote verification. Do not treat CPU CI as CUDA validation.

Next useful milestone: actual free compatible CUDA validation when available. Otherwise audit the original operator's numeric edge coverage and checker sensitivity with independent failure injection, including maximum-row shapes and non-dyadic/near-tie logits; preserve existing algorithms rather than rebuilding them. Avoid endless report/documentation expansion. The all-five final acceptance still depends on genuine GPU evidence or an explicitly reported unvalidated portion.

## November 20, first session

The edge audit found an actual CPU precision defect missed by the original study: a 4096-column row `[0,-0.001f,...]` produced mass error 1.9365674233995378e-5, exceeding the existing 1e-5 limit. A sum-only precision change was insufficient for ascending near-ties. The fix preserves the recurrence and float inputs/logits/output, but uses double normalization exponentials and denominator accumulation. No tolerance change. Historical initial results in the source/README remain explicitly historical.

Added `test_edges.cpp`: analytic near-tie oracle, ascending/descending orders, maximum 1024×4096 shape with non-dyadic near-ties/masks, invariances and independent numerical-checker failure injection; wired strict and sanitizer builds into CPU CI. The CUDA kernel and long-double reference remain unchanged; source-bound reports now require the updated CPU source hash, as intended. No actual GPU reports exist to migrate.

Validation required before publication: original CPU/host/Python suites, new edge regressions, ASAN/UBSAN with leak detection disabled, final diff, clean-archive reruns and remote checks. GPU compilation, device correctness/sanitizers and timings remain pending because nvcc/nvidia-smi are unavailable. Four projects and all three required public demos remain accepted; the fifth is not complete.

Next: actual free compatible GPU validation is still the highest-priority acceptance gate. If unavailable, rotate to a substantive robustness extension in one of the other four new projects rather than endlessly expanding this auxiliary harness or generating cosmetic commits. Preserve old repositories unchanged through November.
