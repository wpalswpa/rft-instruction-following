#!/usr/bin/env bash
# 전체 실험을 순서대로 실행한다: 거부 샘플링 -> LoRA 학습 -> 세 조건 생성 -> 지표 계산.
set -euo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/src"
$PY rft_sample.py --k 4
$PY train_lora.py
for c in B0 B1 M1; do $PY evaluate.py generate --cond $c; done
$PY evaluate.py report
