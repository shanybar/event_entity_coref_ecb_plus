## Document Clustering

* Load the ECB+ test set (the full documents) and save it as a pickle:

`python load_dataset.py --ecb_path <ecb_path> --output_dir <output_dir>`

* Run document clustering script:

`python cluster_topics.py --in_file <test_set_pickle_file> --out_dir <output_dir>`

* Create the predicted topics pickle file from the clustering output:

`python clustering_output_to_topics.py --input_dir <input_dir> --out_dir <output_dir>`

The predicted topics pickle file is then used at the test time.