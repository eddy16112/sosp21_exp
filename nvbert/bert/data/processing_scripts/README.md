# Location of the input files 

This [GCS location](https://console.cloud.google.com/storage/browser/pkanwar-bert) contains the following.
* TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
* Vocab file (vocab.txt) to map WordPiece to word id.
* Config file (bert_config.json) which specifies the hyperparameters of the model.

# Download and preprocess datasets

Download the [wikipedia dump](https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2) and extract the pages
The wikipedia dump can be downloaded from this link in this directory, and should contain the following file:
enwiki-20200101-pages-articles-multistream.xml.bz2

Run [WikiExtractor.py](https://github.com/attardi/wikiextractor) to extract the wiki pages from the XML
The generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has 100 files from wiki_00 to wiki_99, except the last sub directory. For the 20200101 dump, the last file is FE/wiki_17.

Clean up
The clean up scripts (some references here) are in the scripts directory.
The following command will run the clean up steps, and put the results in `./results`
`./process_wiki.sh '<data dir>/*/wiki_??'`

```shell
cd cleanup_scripts  
mkdir -p wiki  
cd wiki  
wget https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2    # Optionally use curl instead  
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2  
cd ..    # back to bert/cleanup_scripts  
git clone https://github.com/attardi/wikiextractor.git  
python3 wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml    # Results are placed in bert/cleanup_scripts/text  
./process_wiki.sh '<text/*/wiki_??'  
```

After running the `process_wiki.sh` script, for the 20200101 wiki dump, there will be 500 files, named `part-00xxx-of-00500` in the `./results` directory.

# Checkpoint conversion
```shell
python convert_tf_checkpoint.py --tf_checkpoint model.ckpt-28252 --bert_config_path bert_config.json --output_checkpoint checkpoint.pt
```

# Generate the BERT input dataset

The `create_pretraining_data.py` script duplicates the input plain text, replaces different sets of words with masks for each duplication, and serializes the output into the HDF5 file format. The following command will create the BERT dataset in HDF5 format, with results in a directory `hdf5`:

```shell
./parallel_create_hdf5.sh
```

The generated HDF5 dataset has 500 parts, totalling to ~539GB.

To simplify this process, instead of running create_pretraining_data.py directly, use the following commands to generate
the HDF5 files in parallel:

```shell
./create_pretraining_data_wrapper.sh results
```

To reshard the dataset, run the following:
```shell
python3 chop_hdf5_files.py
```



