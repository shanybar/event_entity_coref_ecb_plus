
import os
import spacy
import codecs
import logging
import argparse
import numpy as np
import sklearn.cluster
import _pickle as cPickle
from sklearn import metrics

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('--in_file', help='the file containing the test corpus')
ap.add_argument('--out_dir', help='where to save the output cluster files')
ap.add_argument('--filter', action='store_true', default=False, help='filter words that are not verbs or named entities')

args = ap.parse_args()

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

nlp = spacy.load('en')

NUM_TOPICS = 20
N_COMPONENTS = 10000


def filter_docs(docs):
    filtered_docs = {}

    for doc_id, doc in docs.iteritems():
        parsed = nlp(unicode(doc))
        filtered_tokens = [tok.text for tok in parsed if tok.ent_iob_ != 'O']
        if len(filtered_tokens) < 3:
            filtered_doc = doc
        else:
            filtered_doc = ' '.join(filtered_tokens)

        filtered_docs[doc_id] = filtered_doc

    return filtered_docs


def get_sub_topics(doc_id):
    topic = doc_id.split('_')[0]
    if 'ecbplus' in doc_id:
        category = 'ecbplus'
    else:
        category = 'ecb'
    return '{}_{}'.format(topic, category)


def main():
    logger.info('Reading sentences from {}'.format(args.in_file))

    with open(args.in_file, 'rb') as f:
        docs = cPickle.load(f)
    if args.filter:
        docs = filter_docs(docs)

    doc_ids = list(docs.keys())
    sentences = list(docs.values())

    true_labels = [get_sub_topics(doc_id) for doc_id in doc_ids]
    true_clusters_set = set(true_labels)

    labels_mapping = {}
    for label in true_clusters_set:
        labels_mapping[label] = len(labels_mapping)

    true_labels_int = [labels_mapping[label] for label in true_labels]

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, ngram_range=(1,3),
                                 stop_words='english')

    X = vectorizer.fit_transform(sentences)

    print('Number of documents - {}'.format(len(sentences)))

    logger.info('Clustering to topics...')

    kmeans = sklearn.cluster.KMeans(n_clusters=NUM_TOPICS, init='k-means++', max_iter=200,
                                    n_init=20,random_state=665,
                                    n_jobs=20,algorithm='auto')
    kmeans.fit(X)

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # Evaluation
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels_int, kmeans.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(true_labels_int, kmeans.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(true_labels_int, kmeans.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(true_labels_int, kmeans.labels_))

    logger.info('Saving output to {}'.format(out_dir))
    for cluster_id in np.unique(kmeans.labels_):
        curr_sentences = {doc_ids[i]+'_'+sentences[i] for i in np.nonzero(kmeans.labels_ == cluster_id)[0]}
        with codecs.open(os.path.join(out_dir, '{}.txt'.format(cluster_id)), 'w', 'utf-8') as f_out:
            for sentence in curr_sentences:
                f_out.write(sentence + '\n')


    print('Number of documents - {}'.format(len(sentences)))


if __name__ == '__main__':
    main()