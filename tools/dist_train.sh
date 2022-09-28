#!/usr/bin/env bash

CONFIG=$1

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29500 \
  tools/train.py \
  $CONFIG \
  --launcher pytorch \
  --seed 42
