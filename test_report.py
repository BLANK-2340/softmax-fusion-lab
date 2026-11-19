"""MIT. SYNTHETIC UNIT TEST LOGS ONLY. None are real GPU measurements."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark_report as br


def synthetic_log():
    """Artificial parser fixture, NEVER publish it as benchmark evidence."""
    line = ("rows=8 cols=33 warmup=2 repetitions=4 min_ms=1 median_ms=2 mean_ms=2 max_ms=3 "
            "max_abs_error=0.000001 max_mass_error=0.000001")
    return ("device=SYNTHETIC TEST DEVICE sm=7.5 global_bytes=1024 driver=12000 runtime=12000 "
            "nvcc=12.0.0 host_compiler=SYNTHETIC block_threads=256\n" + br.SCOPE + "\n" +
            "variant=materialized " + line + "\nvariant=fused " + line + "\n" + br.SUCCESS + "\n")


def synthetic_report():
    log = synthetic_log()
    return {"format": 1, "kind": "captured-local-process-log-not-hardware-attestation",
            "sources": br.source_hashes(), "binary_sha256": "a"*64, "build_command": "SYNTHETIC TEST COMMAND",
            "returncode": 0, "stdout": log, "stderr": "", "stdout_sha256": br.digest(log.encode()),
            "parsed": br.parse_log(log, "", 0)}


class ReportTests(unittest.TestCase):
    def test_known_fixture_fields(self):
        result = br.validate(synthetic_report())
        self.assertEqual(result["device"], "SYNTHETIC TEST DEVICE")
        self.assertEqual([x["name"] for x in result["variants"]], ["materialized", "fused"])
        self.assertEqual(result["variants"][0]["median_ms"], 2)

    def test_failed_and_incomplete_process(self):
        log = synthetic_log()
        for stdout, stderr, code in ((log, "", 1), (log, "warning", 0), (log, "", True),
                                      ("\n".join(log.splitlines()[:-1]), "", 0), (log+"FAIL\n", "", 0)):
            with self.subTest(code=code, stderr=stderr), self.assertRaises(ValueError):
                br.parse_log(stdout, stderr, code)

    def test_corrupted_measurement_fields(self):
        for old, new in (("min_ms=1", "min_ms=nan"), ("max_ms=3", "max_ms=inf"),
                         ("min_ms=1", "min_ms=-1"), ("mean_ms=2", "mean_ms=4"),
                         ("median_ms=2", "median_ms=0"), ("warmup=2", "warmup=0"),
                         ("repetitions=4", "repetitions=1"), ("cols=33", "cols=4097"),
                         ("max_mass_error=0.000001", "max_mass_error=1"),
                         ("global_bytes=1024", "global_bytes=0"), ("sm=7.5", "sm=unknown"),
                         ("scope=CUDA", "scope=CPU"), ("variant=fused", "variant=materialized")):
            with self.subTest(new=new), self.assertRaises(ValueError):
                br.parse_log(synthetic_log().replace(old, new), "", 0)
        with self.assertRaises(ValueError):
            br.parse_log(synthetic_log().replace("repetitions=4", "repetitions=5", 1), "", 0)

    def test_source_log_and_schema_binding(self):
        r = synthetic_report()
        for key, value in (("format", True), ("kind", "real_gpu_validated"), ("sources", {}),
                           ("binary_sha256", "bad"), ("build_command", ""), ("stdout_sha256", "bad"),
                           ("parsed", {})):
            changed = copy.deepcopy(r)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                br.validate(changed)
        with self.assertRaises(ValueError):
            br.validate({**r, "extra": 1})

    def test_json_bounds_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"fixture.json"
            for value in ('{"format":1,"format":2}', '{"x":NaN}', ' '*(br.LIMIT+1)):
                p.write_text(value)
                with self.assertRaises(ValueError):
                    br.read_report(p)
            p.write_text(json.dumps(synthetic_report()))
            command = [sys.executable, str(Path(br.__file__)), "validate", str(p)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("NOT hardware attestation", result.stdout)
            p.write_text('{}')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("schema mismatch", result.stderr)

    def test_capture_mocked_not_gpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/"SYNTHETIC_NONEXECUTABLE"
            binary.write_bytes(b"unit test only; never executed")
            output = Path(directory)/"SYNTHETIC_REPORT.json"
            fake = subprocess.CompletedProcess([], 0, synthetic_log(), "")
            with patch.object(br.subprocess, "run", return_value=fake) as run:
                br.capture(binary, ["8", "33", "2", "4"], "SYNTHETIC", output)
                self.assertEqual(run.call_args.args[0], [str(binary), "8", "33", "2", "4"])
                self.assertNotIn("shell", run.call_args.kwargs)
                br.validate(br.read_report(output))
                before = output.read_bytes()
                with self.assertRaises(ValueError):
                    br.capture(binary, ["8", "33", "2", "4"], "SYNTHETIC", output)
                self.assertEqual(output.read_bytes(), before)

    def test_failed_capture_does_not_create_report(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/"SYNTHETIC_NONEXECUTABLE"
            binary.write_bytes(b"test")
            output = Path(directory)/"absent.json"
            for fake in (subprocess.CompletedProcess([], 1, synthetic_log(), ""),
                         subprocess.CompletedProcess([], 0, synthetic_log().replace("rows=8", "rows=9"), "")):
                with patch.object(br.subprocess, "run", return_value=fake), self.assertRaises(ValueError):
                    br.capture(binary, ["8", "33", "2", "4"], "SYNTHETIC", output)
                self.assertFalse(output.exists())
            with patch.object(br.subprocess, "run", side_effect=subprocess.TimeoutExpired("synthetic", 180)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    br.capture(binary, ["8", "33", "2", "4"], "SYNTHETIC", output)
            self.assertFalse(output.exists())
            hashes = br.source_hashes()
            with patch.object(br.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, synthetic_log(), "")), \
                    patch.object(br, "source_hashes", side_effect=[hashes, {}]), self.assertRaises(ValueError):
                br.capture(binary, ["8", "33", "2", "4"], "SYNTHETIC", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
