# Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution

## Introduction
This code was used in the paper:

<b>"Revisiting Joint Modeling of Cross-document Entity and Event Coreference Resolution"</b><br/>
Shany Barhom, Vered Shwartz, Alon Eirew, Michael Bugert, Nils Reimers and Ido Dagan. ACL 2019.

A neural model implemented in PyTorch for resolving cross-document entity and event coreference.
The model was trained and evaluated on the ECB+ corpus.

## Prerequisites
* Python 3.6
* [PyTorch](https://pytorch.org/) 0.4.0
    * We specifically used PyTorch 0.4.0 with CUDA 9.0 on Linux, which can be installed with the following:
    `pip install https://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl`
* [spaCy](https://spacy.io/) 2.0.18
* [Matplotlib](https://matplotlib.org/) 3.0.2
* [NumPy](https://www.numpy.org/) 1.16.1
* [NLTK](https://www.nltk.org/) 3.4
* [scikit-learn](https://scikit-learn.org/) 0.20.2
* [SciPy](https://www.scipy.org/) 1.2.1
    * Install the spacy en model with `python -m spacy download en`
* [seaborn] (https://seaborn.pydata.org/) 0.9.0
* [AllenNLP](https://allennlp.org/) 0.5.1

## Testing Instructions
* Download pretrained event and entity models and pre-processed data for the ECB+ corpus at https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK
    * Configure the model and test set paths in the configuration file test_config.json accordingly.
* Run the script predict_model.py with the command:
    `python src/all_models/predict_model.py --config_path test_config.json --out_dir <output_directory>`

Where:
* `config_path` - a path to a JSON file holds the test configuration (test_config.json).
     An explanation about this configuration file is provided in config_files_readme.md.
* `out_dir` - an output directory.

Main output:
* Two response (aka system prediction) files:
   * `CD_test_entity_mention_based.response_conll` - cross-document entity coreference results in CoNLL format.
   * `CD_test_event_mention_based.response_conll` - cross-document event coreference results in CoNLL format.
* `conll_f1_scores.txt` - A text file contains the CoNLL coreference scorer's output (F1 score).

Note - the script's configuration file (test_config.json) also requires: 
   * An output file of a within-document entity coreference system on the ECB+ corpus (provided in this repo at             data/external/stanford_neural_wd_entity_coref_out/ecb_wd_coref.json)
   * An output file of the document clustering algorithm that has been used in the paper (provided in this repo at data/external/document_clustering/predicted_topics)

## Training Instructions
* Download the pre-processed data for the ECB+ corpus at https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK.
    * Alternatively, you can create the data from scratch by following the instructions below.
* Configure the train and dev set paths in the configuration file train_config.json.
* Run the script train_model.py with the command:
   `python src/all_models/train_model.py --config_path train_config.json --out_dir <output_directory>`

Where:
* config_path - a path to a JSON file holds the training configuration (train_config.json).
   An explanation about this configuration file is provided in config_files_readme.md.
* out_dir - an output directory.

Main Output:
* Two trained models that are saved to the files:
    * `cd_event_best_model` - the event model that got the highest B-cubed F1 score on the dev set.
    * `cd_entity_best_model` - the entity model that got the highest B-cubed F1 score on the dev set.
* `summery.txt` - a summary of the training.

Note - the script's configuration file (train_config.json) also requires: 
   * An output file of a within-document entity coreference system on the ECB+ corpus (provided in this repo at             data/external/stanford_neural_wd_entity_coref_out)
 

## Creating Data from Scratch
This repository provides pre-processed data for the ECB+ corpus (download from https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK).
In case you want to create the data from scratch, do the following steps:

* Extract the gold mentions and documents from the ECB+ corpus:
    `python src/data/make_dataset.py --ecb_path <ecb_path> --output_dir <output_directory> --data_setup 2 --selected_sentences_file       data/raw/ECBplus_coreference_sentences.csv`

Where:
* `ecb_path` - a directory contains the ECB+ documents (can be downloaded from http://www.newsreader-project.eu/results/data/the-ecb-corpus/).
* `output_dir` - output directory.
* `data_setup` - enter '2' for loading the ECB+ data in Cybulska setup (see the setup description in the paper).
* `selected_sentences_file` - path to a CSV file contains the selected sentences.

Output:
The script saves for each data split (train/dev/test):
* A json file contains its mention objects.
* A text file contains its sentences.

* Run the feature extraction script, which extracts predicate-argument structures,
  mention head and ELMo embeddings, for each mention in each split (train/dev/test):
    `python src/features/build_features.py --config_path build_features_config.json --output_path <output_path>`

Where:
* `config_path` - a path to a JSON file holds the feature extraction configuration (build_features_config.json).
                  An explanation about this configuration file is provided in config_files_readme.md.
* `output_path` - a path to the output directory

Output:
This script saves 3 pickle files, each contains a Corpus object representing each split:
* `train_data` - the training data, used as an input to the script train_model.py.
* `dev_data` - the dev data, used as an input to the script train_model.py.
* `test_data` - the test data, used as an input to the script predict_model.py.

Note - the script's configuration file also requires:
   * The output files of the script `make_dataset.py` (JSON and text files).
   * Output files of SwiRL SRL system for ECB+ corpus (provided in this repo at data/external/swirl_output).
