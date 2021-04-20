# Requirement
```
conda install pybind11
conda install six
conda install nltk
conda install pyramid
pip install boto3
pip install requests
pip install sentencepiece
install pytorch gpu version https://pytorch.org
install apex: https://github.com/NVIDIA/apex (https://github.com/NVIDIA/apex#linux)
install wikiextractor: https://github.com/attardi/wikiextractor.git
install dllogger https://github.com/NVIDIA/dllogger.git

```

# Megatron-LM
# 1. Install (optional)
```
install megatron-lm
cd megatrom-lm
python setup.py install
```

# 2. pretrain BERT with wikipedia dataset
## Preprocess the dataset (optional, dataset is already preprocessed, named my-bert_text_sentence in the repo)
```
bash megatron18_bert_preprocess_data.sh
```
## Run with the dataset
```
mpirun -np 1 megatron18_bert_pretrain_distributed.sh
```

# 3. pretrain GPT2 with wikipedia dataset
## Preprocess the dataset (optional, dataset is already preprocessed, named my-gpt2_text_sentence in the repo)
```
bash megatron18_bert_preprocess_data.sh
```
## Run with the dataset
```
mpirun -np 1 megatron17_gpt2_pretrain_distributed.sh
```

# 4. evaluate BERT with RACE dataset
## Get the dataset
```
wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
tar -zxf RACE.tar.gz
mv RACE data
```
## Run with the dataset
```
bash megatron_bert_race_eval.sh
```

https://github.com/google-research/bert

# DeepSpeed

# 1. Install
```
pip install deepspeed
```

# 2. Presplit the dataset (optional, dataset is already presplitted, named data/wikipedia/wiki_AA_presplited.json)
```
python presplit_sentences_json.py /scratch2/xluo/program/sosp21_exp/data/wikipedia/wiki_AA.json /scratch2/xluo/program/sosp21_exp/data/wikipedia/wiki_AA_presplited.json
```

# 3. Set the corpora.py
open the DeepSpeed/DeepSpeedExamples/Megatron-LM/data_utils/corpora.py

set PATH = 'data/wikipedia/wiki_AA_presplited.json' 

# 4. Run
```
mpirun -np 1 ./deepspeed_bert_pretrain_mp.sh 
``` 


# Summit module
```
module load open-ce/1.1.3-py37-0
module load gcc/6.4.0 
module load cuda/10.2
```

# Current issue on Summit
## pytorch 1.7, python 3.6,3.7
issue with pytorch 1.7
https://github.com/pytorch/pytorch/issues/47138
nvbert:
RuntimeError: default_program(57): error: identifier "aten_mul_flat__1" is undefined
megatron-lm:
AttributeError: module ‘torch’ has no attribute ‘_amp_foreach_non_finite_check_and_unscale_’

## pytorch 1.7 python 3.8
apex does not find c++ compiler

## megatron-lm v1.1 and deepspeed version
apply this patch for fp32
https://github.com/NVIDIA/Megatron-LM/issues/36
