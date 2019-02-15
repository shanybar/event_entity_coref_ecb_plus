import matplotlib

matplotlib.use('Agg')

import random
import gzip
import codecs
import logging
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt

import argparse
from sklearn.manifold import TSNE
from itertools import cycle, islice

random.seed(1)
np.random.seed(1)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Training a regressor')

parser.add_argument('--embeddings_file', type=str,
                    help=' The path configuration json file')
parser.add_argument('--pdf_out_file', type=str,
                    help=' The directory to the output folder')
parser.add_argument('--type', type=str,
                    help=' The directory to the output folder')

args = parser.parse_args()


def main():
    '''
    This script loads mentions representation components (full representation, context vector,
    and dependent mentions vector), projects the representations and plots each mention in
    a scatter plot (each representation component in a separate plot).
    '''
    embeddings_file = args.embeddings_file
    out_file = args.pdf_out_file

    logger.info('Reading the embeddings from {}...'.format(embeddings_file))
    vocabulary, vocab_cluster_id, wv, gold_to_id = load_representations(args.type)

    logger.info('Computing TSNE...')
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv)

    logger.info('Saving the output file to {}...'.format(out_file))
    fig = matplotlib.pyplot.figure(figsize=(25, 25))

    colors = np.array(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00',"#000000"])

    colorset = [colors[i] for i in vocab_cluster_id]

    ax = plt.axes()
    ax.scatter(Y[:, 0], Y[:, 1], s=700, c=colorset)
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=25)

    fig.savefig(out_file + '.pdf', format='pdf', dpi=2000, bbox_inches='tight')

    logger.info('Done!')


def load_representations(type):
    '''
    This script loads mentions representation component of a specific type
    (full representation/context vector/dependent mentions vector)
    :param type: a string represents the representation component type
    :return: vocab - list of mentions (span and gold cluster id),
            vocab_cluster_id - list of all cluster ids,
            numpy array contains all the vectors,
            and gold_to_id -  mapping of the string represents the gold coref chain
            to a number

    '''
    with open(args.embeddings_file, 'rb') as f:
        mention_to_rep_dict = cPickle.load(f)

    vocab = []
    vocab_cluster_id = []
    wv = []
    cluster_count = 1
    gold_to_id = {}
    gold_clusters = {}
    print(len(mention_to_rep_dict))
    for mention_tuple, rep in mention_to_rep_dict.items():  # first organize in gold clusters
        if mention_tuple[1] not in gold_to_id:
            gold_to_id[mention_tuple[1]] = cluster_count
            cluster_count += 1
        gold_cluster_id = gold_to_id[mention_tuple[1]]
        if gold_cluster_id not in gold_clusters:
            gold_clusters[gold_cluster_id] = []
        gold_clusters[gold_cluster_id].append((mention_tuple, rep))

    gold_clusters_list = [cluster for cluster_id, cluster in gold_clusters.items() if len(cluster) > 5]

    selected_gold_clusters = gold_clusters_list[:10]

    # remove some clusters since they make the visualization looks too loaded
    selected_gold_clusters.pop(1)
    selected_gold_clusters.pop(4)
    selected_gold_clusters.pop(1)

    cluster_id = 0
    for cluster in selected_gold_clusters:
        for mention in cluster:
            mention_tuple = mention[0]
            rep = mention[1]
            vocab.append(mention_tuple[0])

            vocab_cluster_id.append(cluster_id)
            if type == 'full':
                wv.append(rep[0])
            elif type == 'args':
                wv.append(rep[1])
            elif type == 'context':
                wv.append(np.array(rep[2]))
        cluster_id += 1
    return vocab, vocab_cluster_id, np.array(wv),gold_to_id


if __name__ == '__main__':
    main()
