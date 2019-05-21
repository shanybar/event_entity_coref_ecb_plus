import os
import _pickle as cPickle
import logging
import argparse

parser = argparse.ArgumentParser(description=' ')

parser.add_argument('--input_dir', type=str,
                        help=' The directory of the input files')
parser.add_argument('--out_dir', type=str,
                        help=' The directory of the output files')


args = parser.parse_args()


def load_clusters():
    topics = []
    clusters = os.listdir(args.input_dir)
    for cluster_file in clusters:
        doc_names_list = []
        if cluster_file == 'metrics.txt':
            continue
        print(cluster_file)
        full_path = os.path.join(args.input_dir,cluster_file)
        with open(full_path,'r') as f:
            for line in f:
                new_line = line.strip().split('_')
                topic_name = new_line[0]
                doc_name = topic_name + '_'+ new_line[1]
                print(doc_name)
                doc_names_list.append(doc_name)
        topics.append(doc_names_list)

    with open(os.path.join(args.out_dir, 'predicted_topics'), 'wb') as f:
        cPickle.dump(topics, f)


def main():
    load_clusters()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()