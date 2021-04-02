# BERT benchmark on Wikipedia corpus 

This example is a modified version of Nvidia's [BERT example](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)

## Requirement 

Download and pre-process Wikipedia corpus following steps in original [quick start guide](./README_nv.md#quick-start-guide). Then set the `INPUT_DATA` in [submit_pretraining.lsf](./submit_pretraining.lsf) to the data path. 

## How to run 

Simply submit the [job script](./submit_pretraining.lsf) from the example directory. 

The key modifications are for launching Apex data parallel on Summit 
```bash
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}

export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=$head
export MASTER_PORT=29500 # default from torch launcher

echo "Setting env_var RANK=${RANK}"
echo "Setting env_var LOCAL_RANK=${LOCAL_RANK}"
echo "Setting env_var WORLD_SIZE=${WORLD_SIZE}"
echo "Setting env_var MASTER_ADDR=${MASTER_ADDR}"
echo "Setting env_var MASTER_PORT=${MASTER_PORT}"
```
Which is sourced by each rank running the [task script](./scripts/run_pretraining_summit_32node_phase1.sh)  
