#! /bin/bash

# Change for multinode config
MP_SIZE=1

DEBUG=0
if [[ ${DEBUG} == 1 ]];  then
       MP_SIZE=1
       NUM_WORKERS=1
       NUM_GPUS_PER_WORKER=1
       HIDDEN_SIZE=1024
       NUM_ATTN_HEADS=16
       NUM_LAYERS=5
       BATCHSIZE=4
else
       NUM_WORKERS=1
       NUM_GPUS_PER_WORKER=6
       HIDDEN_SIZE=1024
       NUM_ATTN_HEADS=16
       NUM_LAYERS=24
       BATCHSIZE=6

       #HIDDEN_SIZE=4096
       #NUM_LAYERS=24 # 50
       #BATCHSIZE=16
fi


DATA_PATH=my-gpt2_text_document
VOCAB_PATH=gpt2-vocab.json
MERGE_PATH=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
if [[ -z $1 ]]; then
       config_json="ds_zero_stage_2_config.json"

       # offloads to NVMe
       #config_json="$script_dir/ds_zero_stage_infinity_config.json"
else
       config_json=$script_dir/`basename $1`
fi

#ZeRO Configs
stage=3
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=true
CC=true
SYNCHRONIZE=true
PROFILE=false

# TiledLinear splits, 0 is disable
TILED_LINEAR="false"
TILE_DIM=1


# Megatron Model Parallelism
LOGDIR="tboard-zero3/stage${stage}-lazyscatter-${NUM_LAYERS}l_${HIDDEN_SIZE}h_${NUM_WORKERS}n_${NUM_GPUS_PER_WORKER}g_${MP_SIZE}mp_${BATCHSIZE}b"


gpt_options=" \
        --model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --batch-size $BATCHSIZE \
        --train-iters 30 \
        --lr-decay-iters 320000 \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 10 \
        --save-interval 10000 \
        --eval-interval 2000 \
        --eval-iters 10 \
        --scattered-embeddings \
        --split-transformers \
"
        #--tensorboard-dir ${LOGDIR}

 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs}
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

if [ "${TILED_LINEAR}" = "true" ]; then
tile_opt="${tile_opt} \
        --memory-centric-tiled-linear \
        --tile-factor=${TILE_DIM}"
fi


full_options="${gpt_options} ${deepspeed_options} ${tile_opt}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  DeepSpeedExamples/Megatron-LM-v1.1.5-ZeRO3/pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
