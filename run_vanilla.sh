#!/usr/bin/env bash
set -euo pipefail

exec torchrun --standalone --nproc_per_node=8 train_vanilla.py "$@"
