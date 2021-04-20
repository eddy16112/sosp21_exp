#!/bin/bash

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=cn135
MASTER_PORT=6001
NNODES=2
NODE_RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=my-bert_text_sentence
CHECKPOINT_PATH=checkpoints/bert_345m

export RANK=$NODE_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export LOCAL_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
echo "nnodes=${NNODES}"
echo "master=${MASTER_ADDR}"
echo "Setting env_var RANK=${RANK}"
echo "Setting env_var LOCAL_RANK=${LOCAL_RANK}"
echo "Setting env_var WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}"
echo "Setting env_var LOCAL_SIZE=${LOCAL_SIZE}"

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/pretrain_bert.py \
       --local_rank ${LOCAL_RANK} \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 30 \
       --data-path $DATA_PATH \
       --vocab-file bert-large-uncased-vocab.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10

