import os
import sys
import logging
import operator
import collections

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

written_mentions = 0
cd_clusters_count = 10000
wd_clusters_count = 10

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from classes import *


def write_span_based_cd_coref_clusters(corpus, out_file, is_event, is_gold, use_gold_mentions):
    '''
    This function writes the predicted clusters to a file (in a CoNLL format) in a span based manner,
    means that each token is written to the file
    and the coreference chain id is marked in a parenthesis, wrapping each mention span.
    Used in any setup that requires matching of a predicted mentions with gold mentions.
    :param corpus: A Corpus object, contains the documents of each split, grouped by topics.
    :param out_file: filename of the CoNLL output file
    :param is_event: whether to write event or entity mentions
    :param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
    or to write a system file (response) that contains the predicted clusters.
    :param use_gold_mentions: whether to use the gold mentions or predicted mentions
    '''
    mentions_count = 0
    out_coref = open(out_file, 'w')
    ecb_topics = {}
    ecbplus_topics = {}
    cd_coref_chain_to_id = {}
    cd_coref_chain_to_id_counter = 0
    for topic_id, topic in corpus.topics.items():
        if 'plus' in topic_id:
            ecbplus_topics[topic_id] = topic
        else:
            ecb_topics[topic_id] = topic

    topic_keys = sorted(ecb_topics.keys()) + sorted(ecbplus_topics.keys())

    for topic_id in topic_keys:
        curr_topic = corpus.topics[topic_id]
        out_coref.write("#begin document (" + topic_id + "); part 000" + '\n')
        for doc_id in sorted(curr_topic.docs.keys()):
            curr_doc = curr_topic.docs[doc_id]
            for sent_id in sorted(curr_doc.sentences.keys()):
                out_coref.write('\n')
                start_map = collections.defaultdict(list)
                end_map = collections.defaultdict(list)
                word_map = collections.defaultdict(list)
                curr_sent = curr_doc.sentences[sent_id]
                sent_toks = curr_sent.get_tokens()
                if is_event:
                    sent_mentions = curr_sent.gold_event_mentions if use_gold_mentions else curr_sent.pred_event_mentions
                else:
                    sent_mentions  = curr_sent.gold_entity_mentions if use_gold_mentions else curr_sent.pred_entity_mentions
                for mention in sent_mentions:
                    mentions_count += 1
                    if is_gold: # writing key file
                        if mention.gold_tag not in cd_coref_chain_to_id:
                            cd_coref_chain_to_id_counter += 1
                            cd_coref_chain_to_id[mention.gold_tag] = cd_coref_chain_to_id_counter
                        coref_chain = cd_coref_chain_to_id[mention.gold_tag]
                    else: # writing response file
                        coref_chain = mention.cd_coref_chain

                    if use_gold_mentions:
                        start = mention.start_offset
                        end = mention.end_offset
                    else: # ignore predicted mention that doesn't have compatible gold mention (following previous work)
                        if mention.has_compatible_mention: # one should decide which span to use during evaluation (predicted span vs. gold span) since it is not clear what has been done in a previous work (Yang et al. and Choubey et al.)
                            # start = mention.gold_start
                            # end = mention.gold_end
                            start = mention.start_offset
                            end = mention.end_offset
                        else:
                            continue
                    if start == end:
                        word_map[start].append(coref_chain)
                    else:
                        start_map[start].append((coref_chain, end))
                        end_map[end].append((coref_chain, start))

                for k, v in start_map.items():
                    start_map[k] = [cluster_id for cluster_id, end in
                                    sorted(v, key=operator.itemgetter(1), reverse=True)]
                for k, v in end_map.items():
                    end_map[k] = [cluster_id for cluster_id, start in
                                    sorted(v, key=operator.itemgetter(1), reverse=True)]

                for tok in sent_toks:
                    word_index = int(tok.token_id)
                    coref_list = []
                    if word_index in end_map:
                        for coref_chain in end_map[word_index]:
                            coref_list.append("{})".format(coref_chain))
                    if word_index in word_map:
                        for coref_chain in word_map[word_index]:
                            coref_list.append("({})".format(coref_chain))
                    if word_index in start_map:
                        for coref_chain in start_map[word_index]:
                            coref_list.append("({}".format(coref_chain))

                    if len(coref_list) == 0:
                        token_tag = "-"
                    else:
                        token_tag = "|".join(coref_list)

                    out_coref.write('\t'.join([topic_id, '0', tok.token_id, tok.get_token(), token_tag]) + '\n')

        out_coref.write("#end document\n")
        out_coref.write('\n')

    out_coref.close()
    logger.info('{} mentions have been written.'.format(mentions_count))


def write_clusters_to_file(clusters, file_obj,topic):
    '''
    Write the clusters to a text file (used for analysis)
    :param clusters: list of Cluster objects
    :param file_obj: file to write the clusters
    :param topic - topic name
    '''
    i = 0
    file_obj.write('Topic - ' + topic +'\n')
    for cluster in clusters:
        i += 1
        file_obj.write('cluster #' + str(i) + '\n')
        mentions_list = []
        for mention in cluster.mentions.values():
            mentions_list.append('{}_{}'.format(mention.mention_str,mention.gold_tag))
        file_obj.write(str(mentions_list) + '\n\n')


def write_mention_based_cd_clusters(corpus, is_event, is_gold,out_file):
    '''
    This function writes the cross-document (CD) predicted clusters to a file (in a CoNLL format)
    in a mention based manner, means that each token represents a mention and its coreference chain id is marked
    in a parenthesis.
    Used in Cybulska setup, when gold mentions are used during evaluation and there is no need
    to match predicted mention with a gold one.
    :param corpus: A Corpus object, contains the documents of each split, grouped by topics.
    :param out_file: filename of the CoNLL output file
    :param is_event: whether to write event or entity mentions
    :param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
    or to write a system file (response) that contains the predicted clusters.
    '''
    out_coref = open(out_file, 'w')
    cd_coref_chain_to_id = {}
    cd_coref_chain_to_id_counter = 0
    ecb_topics = {}
    ecbplus_topics = {}
    for topic_id, topic in corpus.topics.items():
        if 'plus' in topic_id:
            ecbplus_topics[topic_id] = topic
        else:
            ecb_topics[topic_id] = topic

    generic = 'ECB+/ecbplus_all'
    out_coref.write("#begin document (" + generic + "); part 000" + '\n')
    topic_keys = sorted(ecb_topics.keys()) + sorted(ecbplus_topics.keys())

    for topic_id in topic_keys:
        curr_topic = corpus.topics[topic_id]
        for doc_id in sorted(curr_topic.docs.keys()):
            curr_doc = curr_topic.docs[doc_id]
            for sent_id in sorted(curr_doc.sentences.keys()):
                curr_sent = curr_doc.sentences[sent_id]
                mentions = curr_sent.gold_event_mentions if is_event else curr_sent.gold_entity_mentions
                mentions.sort(key=lambda x: x.start_offset, reverse=True)
                for mention in mentions:
                    # map the gold coref tags to unique ids
                    if is_gold:  # creating the key files
                        if mention.gold_tag not in cd_coref_chain_to_id:
                            cd_coref_chain_to_id_counter += 1
                            cd_coref_chain_to_id[mention.gold_tag] = cd_coref_chain_to_id_counter
                        coref_chain = cd_coref_chain_to_id[mention.gold_tag]
                    else:  # writing the clusters at test time (response files)
                        coref_chain = mention.cd_coref_chain
                    out_coref.write('{}\t({})\n'.format(generic,coref_chain))
    out_coref.write('#end document\n')
    out_coref.close()


def write_mention_based_wd_clusters(corpus, is_event, is_gold, out_file):
    '''
    This function writes the within-document (WD) predicted clusters to a file (in a CoNLL format)
    in a mention based manner, means that each token represents a mention and its coreference chain id is marked
    in a parenthesis.
    Specifically in within document evaluation, we cut all the links across documents, which
    entails evaluating each document separately.
    Used in Cybulska setup, when gold mentions are used during evaluation and there is no need
    to match predicted mention with a gold one.
    :param corpus: A Corpus object, contains the documents of each split, grouped by topics.
    :param out_file: filename of the CoNLL output file
    :param is_event: whether to write event or entity mentions
    :param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
    or to write a system file (response) that contains the predicted clusters.
    '''
    doc_names_to_new_coref_id = {}
    next_doc_increment = 0
    doc_increment = 10000

    out_coref = open(out_file, 'w')
    cd_coref_chain_to_id = {}
    cd_coref_chain_to_id_counter = 0
    ecb_topics = {}
    ecbplus_topics = {}
    for topic_id, topic in corpus.topics.items():
        if 'plus' in topic_id:
            ecbplus_topics[topic_id] = topic
        else:
            ecb_topics[topic_id] = topic

    generic = 'ECB+/ecbplus_all'
    out_coref.write("#begin document (" + generic + "); part 000" + '\n')
    topic_keys = sorted(ecb_topics.keys()) + sorted(ecbplus_topics.keys())

    for topic_id in topic_keys:
        curr_topic = corpus.topics[topic_id]
        for doc_id in sorted(curr_topic.docs.keys()):
            curr_doc = curr_topic.docs[doc_id]
            for sent_id in sorted(curr_doc.sentences.keys()):
                curr_sent = curr_doc.sentences[sent_id]
                mentions = curr_sent.gold_event_mentions if is_event else curr_sent.gold_entity_mentions
                for mention in mentions:
                    # map the gold coref tags to unique ids
                    if is_gold:  # creating the key files
                        if mention.gold_tag not in cd_coref_chain_to_id:
                            cd_coref_chain_to_id_counter += 1
                            cd_coref_chain_to_id[mention.gold_tag] = cd_coref_chain_to_id_counter
                        coref_chain = cd_coref_chain_to_id[mention.gold_tag]
                    else:  # writing the clusters at test time (response files)
                        coref_chain = mention.cd_coref_chain

                    if mention.doc_id not in doc_names_to_new_coref_id:
                        next_doc_increment += doc_increment
                        doc_names_to_new_coref_id[mention.doc_id] = next_doc_increment

                    coref_chain += doc_names_to_new_coref_id[mention.doc_id]

                    out_coref.write('{}\t({})\n'.format(generic,coref_chain))
    out_coref.write('#end document\n')
    out_coref.close()
