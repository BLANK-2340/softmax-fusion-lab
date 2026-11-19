"""MIT, copyright 2026 BLANK-2340. Strict benchmark-log capture/validation.

Report validation is NOT CUDA validation, hardware attestation or proof that a
binary was built from the recorded sources. No GPU measurements ship with this
module. Unit-test logs are explicitly synthetic and never exported as results.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = ("softmax_cuda.cu", "softmax_cpu.cpp", "bench_common.hpp")
LIMIT = 256_000
SCOPE = ("scope=CUDA events on default stream; kernel pipeline only, includes device idle/launch gaps; "
         "excludes allocation, copies, CPU reference, validation and event creation; no end-to-end timing")
SUCCESS = "PASS device numerical suite and benchmark correctness; GPU results apply only to this execution"
META = re.compile(r"device=(.+) sm=([1-9][0-9]*\.[0-9]+) global_bytes=([0-9]+) driver=([0-9]+) "
                  r"runtime=([0-9]+) nvcc=([0-9]+\.[0-9]+\.[0-9]+) host_compiler=(.+) block_threads=256")
FIELDS = ("rows", "cols", "warmup", "repetitions", "min_ms", "median_ms", "mean_ms", "max_ms",
          "max_abs_error", "max_mass_error")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes():
    return {name: digest((ROOT / name).read_bytes()) for name in SOURCES}


def parse_log(stdout, stderr, returncode):
    require(type(returncode) is int and returncode == 0, "benchmark exited unsuccessfully")
    require(type(stdout) is str and type(stderr) is str, "logs must be text")
    require(len(stdout.encode()) + len(stderr.encode()) <= LIMIT, "log too large")
    require(not stderr.strip(), "stderr must be reviewed, not accepted as a clean run")
    lines = stdout.splitlines()
    require(len(lines) == 5, "missing/extra log lines")
    match = META.fullmatch(lines[0])
    require(match is not None, "missing/invalid device or compiler metadata")
    device, sm, memory, driver, runtime, nvcc, host = match.groups()
    require(device.strip() and host.strip(), "empty metadata")
    require(all(int(x) > 0 for x in (memory, driver, runtime)), "invalid hardware/runtime metadata")
    require(lines[1] == SCOPE, "unsupported or altered timing scope")
    require(lines[4] == SUCCESS, "missing final success marker")
    variants = []
    for line, name in zip(lines[2:4], ("materialized", "fused")):
        parts = line.split()
        require(len(parts) == len(FIELDS) + 1 and parts[0] == "variant=" + name, "variant order/count mismatch")
        values = {}
        for part, field in zip(parts[1:], FIELDS):
            require(part.startswith(field + "="), "unexpected/duplicate field")
            raw = part[len(field) + 1:]
            if field in FIELDS[:4]:
                require(re.fullmatch(r"[0-9]{1,4}", raw) is not None, "invalid integer field")
                values[field] = int(raw)
            else:
                values[field] = float(raw)
                require(math.isfinite(values[field]), "nonfinite measurement")
        require(2 <= values["rows"] <= 1024 and 1 <= values["cols"] <= 4096, "shape outside useful benchmark contract")
        require(1 <= values["warmup"] <= 100 and 2 <= values["repetitions"] <= 1000, "insufficient warmup/repetitions")
        lo, med, avg, hi = (values[f] for f in ("min_ms", "median_ms", "mean_ms", "max_ms"))
        require(0 < lo <= med <= hi and lo <= avg <= hi, "inconsistent/zero timing statistics")
        require(0 <= values["max_abs_error"] <= .0011 and 0 <= values["max_mass_error"] <= 1e-5,
                "reported errors outside bounded numerical tolerances")
        variants.append({"name": name, **values})
    require(all(variants[0][f] == variants[1][f] for f in FIELDS[:4]), "unpaired workload/counts")
    return {"device": device, "sm": sm, "global_bytes": int(memory), "driver": int(driver),
            "runtime": int(runtime), "nvcc": nvcc, "host_compiler": host, "block_threads": 256,
            "scope": SCOPE, "variants": variants}


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def read_report(path):
    with open(path, "rb") as stream:
        data = stream.read(LIMIT + 1)
    require(len(data) <= LIMIT, "report too large")
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate(report):
    keys = {"format", "kind", "sources", "binary_sha256", "build_command", "returncode",
            "stdout", "stderr", "stdout_sha256", "parsed"}
    require(type(report) is dict and set(report) == keys, "report schema mismatch")
    require(type(report["format"]) is int and report["format"] == 1, "unsupported format")
    require(report["kind"] == "captured-local-process-log-not-hardware-attestation", "invalid provenance label")
    require(report["sources"] == source_hashes(), "source hash mismatch")
    require(type(report["binary_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", report["binary_sha256"]), "invalid binary digest")
    require(type(report["build_command"]) is str and 0 < len(report["build_command"].strip()) <= 4096, "missing build command")
    parsed = parse_log(report["stdout"], report["stderr"], report["returncode"])
    require(report["stdout_sha256"] == digest(report["stdout"].encode()), "log hash mismatch")
    require(type(report["parsed"]) is dict and
            json.dumps(report["parsed"], sort_keys=True, allow_nan=False) ==
            json.dumps(parsed, sort_keys=True, allow_nan=False), "parsed fields differ from raw log")
    return parsed


def capture(binary, arguments, build_command, destination):
    require(len(arguments) == 4 and all(re.fullmatch(r"[0-9]{1,4}", x) for x in arguments), "four numeric arguments required")
    rows, cols, warmup, repetitions = map(int, arguments)
    require(2 <= rows <= 1024 and 1 <= cols <= 4096 and 1 <= warmup <= 100 and 2 <= repetitions <= 1000,
            "capture requires bounded shape, warmup and repeated measurements")
    require(type(build_command) is str and 0 < len(build_command.strip()) <= 4096, "record the actual build command")
    require(not Path(destination).exists(), "output already exists")
    executable = Path(binary).resolve(strict=True)
    sources = source_hashes()
    binary_hash = digest(executable.read_bytes())
    # No shell and no inherited command text evaluation. Only run the chosen executable.
    run = subprocess.run([str(executable), *arguments], capture_output=True, text=True, timeout=180)
    require(sources == source_hashes() and binary_hash == digest(executable.read_bytes()), "source/binary changed during run")
    parsed = parse_log(run.stdout, run.stderr, run.returncode)
    report = {"format": 1, "kind": "captured-local-process-log-not-hardware-attestation",
              "sources": sources, "binary_sha256": binary_hash, "build_command": build_command,
              "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
              "stdout_sha256": digest(run.stdout.encode()), "parsed": parsed}
    # Ensure the executable actually reported the requested measurement counts.
    require(all([v[f] for f in FIELDS[:4]] == [rows, cols, warmup, repetitions] for v in parsed["variants"]),
            "executed log does not match requested arguments")
    validate(report)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    require(len(encoded.encode()) <= LIMIT, "encoded report too large")
    with open(destination, "x", encoding="utf-8") as stream:
        stream.write(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("report")
    record = commands.add_parser("capture")
    record.add_argument("binary")
    record.add_argument("report")
    record.add_argument("--build-command", required=True)
    record.add_argument("--shape", nargs=4, default=["128", "257", "10", "50"], metavar=("ROWS", "COLS", "WARMUP", "REPS"))
    args = parser.parse_args()
    try:
        if args.command == "validate":
            validate(read_report(args.report))
            print("Report is internally consistent and source-bound; this is NOT hardware attestation or project completion.")
        else:
            capture(args.binary, args.shape, args.build_command, args.report)
            print("Captured a complete process log; device execution still requires independent review.")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
