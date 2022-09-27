#!/usr/bin/env bash

CONFIG=$1


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=29500 \
  tools/train.py \
  $CONFIG \
  --gpus 4 \
  --launcher pytorch \
  --seed 42
