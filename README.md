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

# 2. BERT with RACE dataset
## Get the dataset
```
wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
tar -zxf RACE.tar.gz
mkdir data
mv RACE data
```
## Run with the dataset
```
bash bert_race_eval.sh
```
