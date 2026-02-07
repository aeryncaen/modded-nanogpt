#!/usr/bin/env bash
set -euo pipefail

# Geo pre-bias defaults from Shakespeare search
export GEO_PREBIAS_ENABLE=1
export GEO_PREBIAS_METHOD=kl_bucket
export GEO_PREBIAS_MTP_WEIGHTS=1.0,0.5,0.25
export GEO_PREBIAS_BLEND=0.75
export GEO_PREBIAS_RANK=1

# Bigram-aware geo pre-bias for modded-nanogpt
export GEO_PREBIAS_BIGRAM_ENABLE=1
export GEO_PREBIAS_BIGRAM_BLEND=1.0
export GEO_PREBIAS_BIGRAM_RANK=1
export GEO_BIGRAM_LAYERS_RATIO=0.75
export GEO_BIGRAM_LAMBDA_INIT=0.3125

# Embed LR hold/ramp from search defaults
export GEO_PREBIAS_EMBED_LR_SCALE_INIT=1.0
export GEO_PREBIAS_EMBED_LR_HOLD_STEPS=225
export GEO_PREBIAS_EMBED_LR_RAMP_STEPS=75

exec torchrun --standalone --nproc_per_node=8 train_gpt.py "$@"
