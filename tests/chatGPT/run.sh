#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_FILE="$ROOT/tests/chatGPT/test_ex2_edge_cases.py"
BIN_DIR="$ROOT/bin"

mkdir -p "$BIN_DIR"

PYTHON_BIN=""
if [[ -x "$ROOT/.venv/bin/python" ]] && "$ROOT/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1 && python - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  PYTHON_BIN="python3"
else
  echo "error: no Python interpreter with numpy is available" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <version-suffix> [<version-suffix> ...]" >&2
  echo "example: $0 2 3 rand" >&2
  exit 1
fi

printf '| version | source | copied to | status | summary |\n'
printf '|---|---|---|---|---|\n'

for suffix in "$@"; do
  source_file="$ROOT/my/versions/ex2-v${suffix}.py"
  copied_file="$BIN_DIR/ex2-v${suffix}.py"

  if [[ ! -f "$source_file" ]]; then
    printf '| %s | %s | %s | MISSING | source file not found |\n' "$suffix" "$source_file" "$copied_file"
    continue
  fi

  cp "$source_file" "$copied_file"

  set +e
  output="$(TEST_FILE="$TEST_FILE" EX2_STUDENT_PATH="$copied_file" "$PYTHON_BIN" - <<'PY' 2>&1
from importlib import util
from pathlib import Path
import os


test_path = Path(os.environ["TEST_FILE"])
spec = util.spec_from_file_location("edge_tests", test_path)
module = util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

tests = [
    (name, getattr(module, name))
    for name in sorted(dir(module))
    if name.startswith("test_") and callable(getattr(module, name))
]

passed = 0
failed = 0

for name, test in tests:
    try:
        test()
    except Exception:
        failed += 1
    else:
        passed += 1

summary = []
if passed:
    summary.append(f"{passed} passed")
if failed:
    summary.append(f"{failed} failed")

print(", ".join(summary) if summary else "0 tests")
raise SystemExit(0 if failed == 0 else 1)
PY
)"
  status=$?
  set -e

  summary="$(printf '%s\n' "$output" | tail -n 1)"
  if [[ $status -eq 0 ]]; then
    result="PASS"
  else
    result="FAIL"
  fi

  summary="${summary//|/\\|}"
  printf '| %s | %s | %s | %s | %s |\n' "$suffix" "$source_file" "$copied_file" "$result" "$summary"
done