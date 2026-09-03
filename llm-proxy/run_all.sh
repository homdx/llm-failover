#!/bin/sh
# Run every standalone check against a proxy folder.
#
#     ./run_all.sh ../llm-proxy
#
# The folder must hold the proxy script (main.py or python_proxy2.py) and
# its config.toml. Nothing touches the network; every upstream is faked
# in-process. Exit status is 0 only if all of them pass.
set -e

if [ -z "$1" ]; then
    echo "usage: $0 <folder-with-the-proxy>" >&2
    exit 2
fi

TARGET=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
FAILED=0

for t in test_truncation test_false_positives test_coordination test_aborted_response; do
    echo
    echo "=============================================================="
    echo "  $t"
    echo "=============================================================="
    if ! python3 "$HERE/$t.py" "$TARGET"; then
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "=============================================================="
echo "  bench_ratelimit_storm (measurement, not pass/fail)"
echo "=============================================================="
python3 "$HERE/bench_ratelimit_storm.py" "$TARGET" || true

echo
if [ "$FAILED" -eq 0 ]; then
    echo "all checks passed"
else
    echo "$FAILED script(s) reported failures"
fi
exit "$FAILED"
