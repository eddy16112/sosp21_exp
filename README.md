# Megatron-LM
# 1. Requirement
```
conda install pybind11
conda install six
conda install nltk
conda install pyramid
pip install boto3
pip install requests
install pytorch gpu version https://pytorch.org
install apex: https://github.com/NVIDIA/apex (https://github.com/NVIDIA/apex#linux)
install wikiextractor: https://github.com/attardi/wikiextractor.git
install dllogger https://github.com/NVIDIA/dllogger.git

install megatron-lm
cd megatrom-lm
python setup.py install
```

# 2. pretrain BERT with wikipedia dataset
## Preprocess the dataset
```
bash megatron_bert_preprocess_data.sh
```
## Run with the dataset
```
bash megatron_pretrain_bert.sh
```

# 3. evaluate BERT with RACE dataset
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
