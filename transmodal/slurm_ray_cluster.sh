#!/bin/bash

REDIS_PASSWORD=$(uuidgen)
export REDIS_PASSWORD

NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES_ARRAY=($NODES)

HEAD_NODE=${NODES_ARRAY[0]}
IP=$(srun --nodes=1 --ntasks=1 -w $HEAD_NODE hostname --ip-address)

if [[ $IP == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$IP"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    IP=${ADDR[1]}
  else
    IP=${ADDR[0]}
  fi
  echo "IPV6 address detected."
fi

PORT=6379
IP_HEAD_ADDR=$IP:$PORT
export IP_HEAD_ADDR

echo "Ray Cluster head at $HEAD_NODE"
srun --nodes=1 --ntasks=1 -w $HEAD_NODE \
  ray start --head --node-ip-address=$IP --port=$PORT --redis-password=$REDIS_PASSWORD --block &

NUM_WORKERS=$(($SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= $NUM_WORKERS; i++)); do
  WORKER=${NODES_ARRAY[$i]}
  echo "Ray Cluster worker $i at $WORKER"
  srun --nodes=1 --ntasks=1 -w $WORKER ray start --address $IP_HEAD_ADDR --redis-password=$REDIS_PASSWORD --block &
done