#!/usr/bin/env bash
# 확장 실험(blueprint 10절): E1 시드 반복·선택 규칙 비교 -> E2 IFEval. 기존 표본(data/rft_samples.jsonl)을 쓴다.
set -euo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/src"
for run in "S 43" "S 44" "R 42" "R 43" "R 44"; do
  set -- $run
  [ -f ../results/ext/gen_$1_s$2.jsonl ] && { echo "skip $1 s$2 (이미 끝남)"; continue; }
  $PY extend.py select --rule $1 --seed $2
  $PY train_lora.py --data data/ext/train_$1_s$2.jsonl --out outputs/ext/$1_s$2 --seed $2 --log results/ext/train_$1_s$2.json
  $PY evaluate.py generate --cond M1 --adapter outputs/ext/$1_s$2 --out results/ext/gen_$1_s$2.jsonl
done
$PY extend.py report
for c in M1 B1 B0; do
  [ -f ../results/ifeval/gen_$c.jsonl ] && { echo "skip ifeval $c (이미 끝남)"; continue; }
  $PY ifeval_eval.py generate --cond $c
done
$PY ifeval_eval.py report
