python megatron-lm/tools/preprocess_data.py \
       --input data/wikipedia/wiki_AA.json \
       --output-prefix my-bert \
       --vocab bert-large-uncased-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
