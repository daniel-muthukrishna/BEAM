#!/usr/bin/env bash
cd ~/BEAM || exit 1
N="${1:-2}"
mkdir -p logs
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
for (( i=0; i<N; i++ )); do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/generate.py \
        --config configs/generation_config.yaml \
        --device cuda:0 --num-shards "$N" --shard-id "$i" \
        2>&1 | tee "logs/shard$i.log" &
done
wait