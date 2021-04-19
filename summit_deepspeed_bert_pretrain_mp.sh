#!/bin/bash

nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}
summit_nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

GPUS_PER_NODE=6
# Change for multinode config
MASTER_ADDR=$head
MASTER_PORT=29502
NNODES=$summit_nnodes
#NNODES=4
NODE_RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

MP=$1
BATCH_SIZE=$2

export RANK=$NODE_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_ADDR=$head
export MASTER_PORT=29502
echo "nnodes=${NNODES}"
echo "master=${MASTER_ADDR}"
echo "Setting env_var RANK=${RANK}"
echo "Setting env_var LOCAL_RANK=${LOCAL_RANK}"
echo "Setting env_var WORLD_SIZE=${WORLD_SIZE}"
echo "LOCAL_SIZE=${LOCAL_SIZE}"
echo "MP=${MP}"
echo "BATCH_SIZE=${BATCH_SIZE}"

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python DeepSpeed/DeepSpeedExamples/Megatron-LM/pretrain_bert.py \
       --local_rank ${LOCAL_RANK} \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 30 \
       --log-interval 10 \
       --resume-dataloader \
       --train-data wikipedia \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01

