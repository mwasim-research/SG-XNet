#!/usr/bin/env bash
# Unzip the released result files into the folders the analysis scripts expect.
set -euo pipefail
cd "$(dirname "$0")/.."
for z in results_archive/*.zip; do
  echo "unpacking $z"
  unzip -q -n "$z" -d .
done
echo "done: results/ and results_posthoc_*/ are ready"
