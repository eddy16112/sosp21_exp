# Megatron-LM
# 1. Requirement
```
conda install pybind11
conda install six
conda install regex
conda install nltk
conda install pyramid
install apex: https://github.com/NVIDIA/apex
#python setup.py install
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
