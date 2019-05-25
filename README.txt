Please go over the module_versions.txt file and the config_files_readme.txt before running any code.

1) Extraction of gold mentions and documents from the ECB+ corpus:
------------------------------------------------------------------
python src/data/make_dataset.py --ecb_path data/raw/ECB+ --output_dir data/interim/... --data_setup 2 --selected_sentences_file data/raw/ECBplus_coreference_sentences.csv

    Parameters:
        ecb_path - a directory contains the ECB+ documents, ordered in folders by their gold topics.
        output_dir - output directory
        data_setup - 2 for Cybulska setup (recommended) and 1 for Yang setup (not fully supported)
        selected_sentences_file - path to CSV file contains the selected sentences (Cybulska setup)

    Output:
        The script saves for each data split (train/dev/test):
        1) A json file contains its mention objects.
        2) text file contains its sentences.

2) Feature extraction (predicate-argument structures, mention head
   and ELMo embeddings for each mention) for each split (train/dev/test) in Cybulska setup:
-------------------------------------------------------------------------------------------
python src/features/build_features.py --config_path build_features_config.json --output_path data/processed/...

    Parameters:
        config_path - a path to a JSON file holds the feature extraction configuration (build_features_config.json).
                        An explanation about this configuration file is provided in config_files_readme.txt.
        output_path - a path to the output directory

    Output:
        The script saves 3 pickle files, each contains a Corpus object representing the split,
        and can be used for running the train_model.py and the test_model.py scripts.

    Notes:
        Requires the output file/files of SwiRL SRL system

3) Train cross-document event and entity coreference models:
------------------------------------------------------------
python src/all_models/train_model.py --config_path train_config.json --out_dir models/...

     Parameters:
        config_path - a path to a JSON file holds the training configuration (train_config.json)
                      An explanation about this configuration file is provided in config_files_readme.txt.
        out_dir - an output directory

     Output:
        Two models saved to files:
            1) cd_event_best_model - the event model that got the highest B-cubed F1 score on the dev set
            2) cd_entity_best_model - the entity model that got the highest B-cubed F1 score on the dev set
            3) summery.txt - a summary of the training.

4) Test cross-document event and entity coreference models:
-----------------------------------------------------------
python src/all_models/predict_model.py --config_path test_config.json --out_dir models/...

        Parameters:
            config_path - a path to a JSON file holds the test configuration (test_config.json)
                          An explanation about this configuration file is provided in config_files_readme.txt.
            out_dir - an output directory

        Main output:
                1) Two response (aka system prediction) files - CD_test_entity_mention_based.response_conll and CD_test_event_mention_based.response_conll.
                 One for entity coreference and another on for event coreference.
                 Both of them are in a CoNLL format (suitable as an input to CoNLL scorer).
                2) conll_f1_scores.txt - A text file contains the CoNLL coreference scorer's output (F1 score).

