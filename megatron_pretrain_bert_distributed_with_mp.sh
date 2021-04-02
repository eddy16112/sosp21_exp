#!/bin/bash

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=cn142
MASTER_PORT=6000
NNODES=$OMPI_COMM_WORLD_SIZE
NODE_RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "Setting env_var RANK=${NODE_RANK}"
echo "Setting env_var NNODES=${NNODES}"

DATA_PATH=my-bert_text_sentence
CHECKPOINT_PATH=checkpoints/bert_345m

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       megatron-lm/pretrain_bert.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
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
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
