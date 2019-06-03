## Configuration files
In this system, experiments are configured using JSON files.
This readme explains the main attributes of three JSON configuration files:
* build_features_config.json
* train_config.json
* test_config.json

## Configuration file for feature extraction (build_features_config.json):

Used with the script src/features/build_features.py .
To extract features for event and entity mentions, it requires two types of input files
for each split (train/dev/test):
* A json file contains its mention objects (e.g. `train_event_mentions`).
* text file contains its sentences (e.g. `train_text_file`).

Notes:
* The provided build_features_config.json file is configured to extract joint features for event
and entity mentions (with predicate-argument structures extraction).
* SwiRL system's output on the ECB+ corpus is provided with this repo (its directory should be assigned to the srl_output_path attribute).
* ELMo's files (options_file, weight_file) can be downloaded from - *https://allennlp.org/elmo* (we used Original 5.5B model files).

## Configuration file for training (train_config.json):

Used with the script src/all_models/train_model.py.
The provided `train_config.json` file is configured to train joint model for cross-document entity and event coreference.

Most of the attributes are self-explained (e.g. batch_size and lr) , but there are few who need
to be explained:
* `char_pretrained_path/char_vocab_path` - initial character embeddings (provided in this repo at data/external/char_embed). 
    The original embeddings are available at *https://github.com/minimaxir/char-embeddings*.
* `char_rep_size` - the character LSTM's hidden size.
* `feature_size` - embedding size of binary features.
* `glove_path` - path to pre-trained word embeddings. We used glove.6B.300d which can be downloaded from *https://nlp.stanford.edu/projects/glove/*.
* `train_path/dev_path` - path to the pickle files of the train/dev sets, created by the build_features script (and can be downloaded from *https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK*).
* `dev_th_range` - threshold range to tune on the validation set.
* `entity_merge_threshold/event_merge_threshold` - merge threshold during training (for entities/events).
* `merge_iters` -  for how many iterations to run the agglomerative clustering step (during both training and testing). We used 2 iterations.
* `patient` - for how many epochs we allow the model continue training without an improvement on the dev set.
* `use_args_feats` - whether to use argument/predicate vectors.
* `use_binary_feats` -  whether to use the coreference binary features.
* `wd_entity_coref_file` - a path to a file (provided in this repo) which contains the predictions of a WD entity coreference system on the ECB+. We used CoreNLP for that purpose.


## Configuration file for testing (test_config.json):

Used with the script src/all_models/predict_model.py .
The provided test_config.json file is configured to test the joint model for cross-document entity and event coreference.

The main attributes of this configuration files are:
* `test_path` - path to the pickle file of the test set, created by the build_features script (and can be downloaded from *https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK*).
* `cd_event_model_path` - path to the tested event model file.
* `cd_entity_model_path` - path to the tested entity model file.
* `event_merge_threshold/entity_merge_threshold` - merge threshold during testing, tuned on the dev set.
* `use_args_feats`- whether to use argument/predicate vectors.
* `use_binary_feats` -  whether to use the coreference binary features.
* `wd_entity_coref_file` - a path to a file (provided) which contains the predictions of a WD entity coreference system on the ECB+. We use CoreNLP for that purpose.
* `event_gold_file_path` - path to the key (gold) event coreference file (for running the evaluation with the CoNLL scorer), provided in this repo.
* `entity_gold_file_path` - path to the key (gold) entity coreference file (for running the evaluation with the CoNLL scorer), provided in this repo.
* `predicted_topics_path` - path to a pickle file which contains the predicted topics, provided in this repo at data/external/document_clustering or can be obtained using the code in the folder src/doc_clustering.
* `wd_entity_coref_file` - a path to a file (provided in this repo) which contains the predictions of a WD entity coreference system on the ECB+. We used CoreNLP for that purpose.


