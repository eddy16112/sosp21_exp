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

issue with pytorch 1.7
https://github.com/pytorch/pytorch/issues/47138

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
