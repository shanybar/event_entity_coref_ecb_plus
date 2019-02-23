import os
import sys
import json
import spacy
import torch
import random
import logging
import itertools
import collections
import numpy as np
from scorer import *
from eval_utils import *
import _pickle as cPickle
from bcubed_scorer import *
import matplotlib.pyplot as plt
from spacy.lang.en import English

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

from classes import *

clusters_count = 1

analysis_pair_dict = {}


def get_topic(id):
    '''
    Extracts the topic id from the document ID.
    Note that this function doesn't extract the sub-topic ID (including the ecb/ecbplus notation)
    :param id: document id (string)
    :return: the topic id (string)
    '''
    return id.split('_')[0]


def merge_sub_topics_to_topics(test_set):
    '''
    Merges the test's sub-topics sub-topics to their topics (for experimental use).
    :param test_set: A Corpus object represents the test set
    :return: a dictionary contains the merged topics
    '''
    new_topics = {}
    topics_keys = test_set.topics.keys()
    for topic_id in topics_keys:
        topic = test_set.topics[topic_id]
        if get_topic(topic_id) not in new_topics:
            new_topics[get_topic(topic_id)] = Topic(get_topic(topic_id))
        new_topics[get_topic(topic_id)].docs.update(topic.docs)

    return new_topics


def load_predicted_topics(test_set, config_dict):
    '''
    Loads the document clusters that were predicted by a document clustering algorithm and
    organize the test's documents to topics according to document clusters.
    :param test_set: A Corpus object represents the test set
    :param config_dict: a configuration dictionary that contains a path to a file
    stores the results of the document clustering algorithm.
    :return:  a dictionary contains the documents ordered according to the predicted topics
    '''
    new_topics = {}
    with open(config_dict["predicted_topics_path"], 'rb') as f:
        predicted_topics = cPickle.load(f)
    all_docs = []
    for topic in test_set.topics.values():
        all_docs.extend(topic.docs.values())

    all_doc_dict = {doc.doc_id:doc for doc in all_docs }

    topic_counter = 1
    for topic in predicted_topics:
        topic_id = str(topic_counter)
        new_topics[topic_id] = Topic(topic_id)

        for doc_name in topic:
            print(topic_id)
            print(doc_name)
            if doc_name in all_doc_dict:
                new_topics[topic_id].docs[doc_name] = all_doc_dict[doc_name]
        topic_counter += 1

    print(len(new_topics))
    return new_topics


def topic_to_mention_list(topic, is_gold):
    '''
    Gets a Topic object and extracts its event/entity mentions (depends on the is_event flag)
    :param topic: a Topic object
    :param is_event: a flag that denotes whether event mentions will be extracted or
    entity mention will be extracted (True for event extraction, False for entity extraction)
    :param is_gold: a flag that denotes whether to extract gold mention or predicted mentions
    :return: list of the topic's mentions (EventMention or EntityMention objects)
    '''
    event_mentions = []
    entity_mentions = []
    for doc_id, doc in topic.docs.items():
        for sent_id, sent in doc.sentences.items():
            if is_gold:
                event_mentions.extend(sent.gold_event_mentions)
                entity_mentions.extend(sent.gold_entity_mentions)
            else:
                event_mentions.extend(sent.pred_event_mentions)
                entity_mentions.extend(sent.pred_entity_mentions)

    return event_mentions, entity_mentions


def load_entity_wd_clusters(config_dict):
    '''
    Loads from a file the within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    model/tool and ordered those clusters in a dictionary according to their documents.
    :param config_dict: a configuration dictionary that contains a path to a file stores the
    within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    system.
    :return: a dictionary contains a mapping of a documents to their predicted entity clusters
    '''
    doc_to_entity_mentions = {}

    with open(config_dict["wd_entity_coref_file"], 'r') as js_file:
        js_mentions = json.load(js_file)

    # load all entity mentions in the json
    for js_mention in js_mentions:
        doc_id = js_mention["doc_id"].replace('.xml', '')
        if doc_id not in doc_to_entity_mentions:
            doc_to_entity_mentions[doc_id] = {}
        sent_id = js_mention["sent_id"]
        if sent_id not in doc_to_entity_mentions[doc_id]:
            doc_to_entity_mentions[doc_id][sent_id] = []
        tokens_numbers = js_mention["tokens_numbers"]
        mention_str = js_mention["tokens_str"]

        try:
            coref_chain = js_mention["coref_chain"]
        except:
            continue

        doc_to_entity_mentions[doc_id][sent_id].append((doc_id, sent_id, tokens_numbers,
                                                        mention_str, coref_chain))
    return doc_to_entity_mentions


def init_entity_wd_clusters(entity_mentions, doc_to_entity_mentions):
    '''
    Matches entity mentions with their predicted within-document coreference clusters
    produced by an external within-document entity coreference system and forms the initial
    within-document entity coreference clusters.

    :param entity_mentions: gold entity mentions (currently doesn't support
     predicted entity mentions).
    :param doc_to_entity_mentions: a dictionary contains a mapping of a documents to
    their predicted entity clusters.
    :return: a list of Cluster objects contains the initial within-document entity coreference
    clusters
    '''

    doc_to_clusters = {}
    all_entity_clusters = {}

    for entity in entity_mentions:
        doc_id = entity.doc_id
        sent_id = entity.sent_id
        is_entity_found = False
        found_entity = None
        if doc_id in doc_to_entity_mentions and sent_id in doc_to_entity_mentions[doc_id]:
            predicted_entity_mentions = doc_to_entity_mentions[doc_id][sent_id]

            for pred_entity in predicted_entity_mentions:
                pred_start = pred_entity[2][0]
                pred_end = pred_entity[2][-1]
                pred_str = pred_entity[3]
                if have_string_match(entity,pred_str ,pred_start, pred_end):
                    is_entity_found = True
                    found_entity = pred_entity
                    break

        if is_entity_found:
            if doc_id not in doc_to_clusters:
                doc_to_clusters[doc_id] = {}
            pred_coref_chain = found_entity[4]
            if pred_coref_chain not in doc_to_clusters[doc_id]:
                doc_to_clusters[doc_id][pred_coref_chain] = Cluster(is_event=False)
            doc_to_clusters[doc_id][pred_coref_chain].mentions[entity.mention_id] = entity
        else:
            doc_id = entity.doc_id
            if doc_id not in all_entity_clusters:
                all_entity_clusters[doc_id] = []

            singleton = Cluster(is_event=False)
            singleton.mentions[entity.mention_id] = entity
            all_entity_clusters[doc_id].append(singleton)

    count_matched_clusters = 0

    for doc_id, doc_coref_chains in doc_to_clusters.items():
        for coref_chain, cluster in doc_coref_chains.items():
            if doc_id not in all_entity_clusters:
                all_entity_clusters[doc_id] = []
            all_entity_clusters[doc_id].append(cluster)
            if len(cluster.mentions.values()) > 1:
                count_matched_clusters += 1

    print('Matched non-singleton clusters {}'.format(count_matched_clusters))
    logging.info('Matched non-singleton clusters {}'.format(count_matched_clusters))

    return all_entity_clusters


def have_string_match(mention, pred_str, pred_start, pred_end):
    '''
    Checks whether a mention has a match (strict or relaxed) with a predicted mention.
    Used when initializing within-document (WD) entity coreference clusters with the output of
    an external (WD) coreference system.
    :param mention: an EntityMention object
    :param pred_str: predicted mention's text (string)
    :param pred_start: predicted mention's start offset
    :param pred_end: predicted mention's end offset
    :return: True if a match has been found and false otherwise.
    '''
    if mention.mention_str == pred_str and mention.start_offset == pred_start:
        return True
    if mention.mention_str == pred_str:
        return True
    if mention.start_offset >= pred_start and mention.end_offset <= pred_end:
        return True
    if pred_start >= mention.start_offset and pred_end <= mention.end_offset:
        return True

    return False


def init_wd(mentions, is_event):
    '''
    Initialize a set of Mention objects (either EventMention or EntityMention) to a set of
    within-document singleton clusters (a cluster which contains a single mentions), ordered by the mention's
     document ID.
    :param mentions:  a set of Mention objects (either EventMention or EntityMention)
    :param is_event: whether the mentions are event or entity mentions.
    :return: a dictionary contains initial singleton clusters, ordered by the mention's
     document ID.
    '''
    wd_clusters = {}
    for mention in mentions:
        mention_doc_id = mention.doc_id
        if mention_doc_id not in wd_clusters:
            wd_clusters[mention_doc_id] = []
        cluster = Cluster(is_event=is_event)
        cluster.mentions[mention.mention_id] = mention
        wd_clusters[mention_doc_id].append(cluster)

    return wd_clusters


def init_cd(mentions, is_event):
    '''
    Initialize a set of Mention objects (either EventMention or EntityMention) to a set of
    cross-document singleton clusters (a cluster which contains a single mentions).
    :param mentions:  a set of Mention objects (either EventMention or EntityMention)
    :param is_event: whether the mentions are event or entity mentions.
    :return: a list contains initial cross-document singleton clusters.
    '''
    clusters = []
    for mention in mentions:
        cluster = Cluster(is_event=is_event)
        cluster.mentions[mention.mention_id] = mention
        clusters.append(cluster)

    return clusters


def load_embeddings(embed_path, vocab_path):
    '''
    load embeddings from a binary file and a file contains the vocabulary.
    :param embed_path: path to the embeddings' binary file
    :param vocab_path: path to the vocabulary file
    :return: word_embeds - a numpy array containing the word vectors, vocab - a list containing the
    vocabulary.
    '''
    with open(embed_path,'rb') as f:
        word_embeds = np.load(f)

    vocab = []
    for line in open(vocab_path, 'r'):
        vocab.append(line.strip())

    return word_embeds, vocab


def load_one_hot_char_embeddings(char_vocab_path):
    '''
    Loads character vocabulary and creates one hot embedding to each character which later
    can be used to initialize the character embeddings (experimental)
    :param char_vocab_path: a path to the vocabulary file
    :return: char_embeds - a numpy array containing the char vectors, vocab - a list containing the
    vocabulary.
    '''
    vocab = []
    for line in open(char_vocab_path, 'r'):
        vocab.append(line.strip())

    char_to_ix = {}
    for char in vocab:
        char_to_ix[char] = len(char_to_ix)

    char_to_ix[' '] = len(char_to_ix)
    char_to_ix['<UNK>'] = len(char_to_ix)

    char_embeds = np.eye(len(char_to_ix))

    return char_embeds, char_to_ix


def is_stop(w):
    '''
    Checks whether w is a stop word according to a small list of stop words.
    :param w: a word (string)
    :return: True is w is a stop word and false otherwise.
    '''
    return w.lower() in ['a', 'an', 'the', 'in', 'at', 'on','for','very']


def clean_word(word):
    '''
    Removes apostrophes before look for a word in the word embeddings vocabulary.
    :param word: a word (string)
    :return: the word (string) after removing the apostrophes.
    '''
    word = word.replace("'s",'').replace("'",'').replace('"','')
    return word


def get_char_embed(word, model, device):
    '''
    Runs a character LSTM over a word/phrase and returns the LSTM's output vector
    :param word: a word/phrase (string)
    :param model: CDCorefScorer object
    :param device: Pytorch device (gpu/cpu)
    :return:  the character-LSTM's last output vector
    '''
    char_vec = model.get_char_embeds(word, device)

    return char_vec


def find_word_embed(word, model, device):
    '''
    Given a word (string), this function fetches its word embedding (or unknown embeddings for
    OOV words)
    :param word: a word (string)
    :param model: CDCorefScorer object
    :param device: Pytorch device (gpu/cpu)
    :return: a word vector
    '''
    word_to_ix = model.word_to_ix
    word = clean_word(word)
    if word in word_to_ix:
        word_ix = [word_to_ix[word]]
    else:
        lower_word = word.lower()
        if lower_word in word_to_ix:
            word_ix = [word_to_ix[lower_word]]
        else:
            word_ix = [word_to_ix['unk']]

    word_tensor = model.embed(torch.tensor(word_ix,dtype=torch.long).to(device))

    return word_tensor


def find_mention_cluster(mention_id, clusters):
    '''
    Given a mention ID, the function fetches its current predicted cluster.
    :param mention_id: mention ID
    :param clusters: current clusters, should be of the same type (event/entity) as the mention.
    :return: the mention's current predicted cluster
    '''
    for cluster in clusters:
        if mention_id in cluster.mentions:
            return cluster
    raise ValueError('Can not find mention cluster!')


def is_system_coref(mention_id_1, mention_id_2, clusters):
    '''
    Checks whether two mentions are in the same predicted (system) clusters in the current
    clustering configuration.
    :param mention_id_1: first mention ID
    :param mention_id_2: second menton ID
    :param clusters: current clustering configuration (should be of the same type as the mentions,
    e.g. if mention_1 and mention_2 are event mentions, so clusters should be the current event
    clusters)
    :return: True if both mentions belong to the same cluster and false otherwise
    '''
    cluster_1 = find_mention_cluster(mention_id_1, clusters)
    cluster_2 = find_mention_cluster(mention_id_2, clusters)

    if cluster_1 == cluster_2:
        return True
    return False


def create_args_features_vec(mention_1, mention_2 ,entity_clusters, device, model):
    '''
    Creates a vector for four binary features (one for each role - Arg0/Arg1/location/time)
    indicate whether two event mentions share a coreferrential argument in the same role.
    :param mention_1: EventMention object
    :param mention_2: EventMention object
    :param entity_clusters: current entity clusters
    :param device: Pytorch device (cpu/gpu)
    :param model: CDCorefScorer object
    :return: a vector for four binary features embedded as a tensor of size (1,200),
    each feature embedded as 50 dimensional embedding.

    '''
    coref_a0 = 0
    coref_a1 = 0
    coref_loc = 0
    coref_tmp = 0

    if coref_a0 == 0 and mention_1.arg0 is not None and mention_2.arg0 is not None:
        if is_system_coref(mention_1.arg0[1], mention_2.arg0[1],entity_clusters):
            coref_a0 = 1
    if coref_a1 == 0 and mention_1.arg1 is not None and mention_2.arg1 is not None:
        if is_system_coref(mention_1.arg1[1], mention_2.arg1[1],entity_clusters):
            coref_a1 = 1
    if coref_loc == 0 and mention_1.amloc is not None and mention_2.amloc is not None:
        if is_system_coref(mention_1.amloc[1], mention_2.amloc[1],entity_clusters):
            coref_loc = 1
    if coref_tmp == 0 and mention_1.amtmp is not None and mention_2.amtmp is not None:
        if is_system_coref(mention_1.amtmp[1], mention_2.amtmp[1],entity_clusters):
            coref_tmp = 1

    arg0_tensor = model.coref_role_embeds(torch.tensor(coref_a0,
                                                       dtype=torch.long).to(device)).view(1,-1)
    arg1_tensor = model.coref_role_embeds(torch.tensor(coref_a1,
                                                       dtype=torch.long).to(device)).view(1,-1)
    amloc_tensor = model.coref_role_embeds(torch.tensor(coref_loc,
                                                        dtype=torch.long).to(device)).view(1,-1)
    amtmp_tensor = model.coref_role_embeds(torch.tensor(coref_tmp,
                                                        dtype=torch.long).to(device)).view(1,-1)

    args_features_tensor = torch.cat([arg0_tensor,arg1_tensor, amloc_tensor,amtmp_tensor],1)

    return args_features_tensor


def create_predicates_features_vec(mention_1, mention_2, event_clusters, device, model):
    '''
    Creates a vector for four binary features (one for each role - Arg0/Arg1/location/time)
    indicate whether two entity mentions share a coreferrential predicate in the same role.
    :param mention_1: EntityMention object
    :param mention_2: EntityMention object
    :param event_clusters: current entity clusters
    :param device: Pytorch device (cpu/gpu)
    :param model: CDCorefScorer object
    :return: a vector for four binary features embedded as a tensor of size (1,200),
    each feature embedded as 50 dimensional embedding.

    '''
    coref_pred_a0 = 0
    coref_pred_a1 = 0
    coref_pred_loc = 0
    coref_pred_tmp = 0

    predicates_dict_1 = mention_1.predicates
    predicates_dict_2 = mention_2.predicates
    for predicate_id_1, rel_1 in predicates_dict_1.items():
        for predicate_id_2, rel_2 in predicates_dict_2.items():
            if coref_pred_a0 == 0 and rel_1 == 'A0' and rel_2 == 'A0':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_a0 = 1
            if coref_pred_a1 == 0 and rel_1 == 'A1' and rel_2 == 'A1':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_a1 = 1
            if coref_pred_loc == 0 and rel_1 == 'AM-LOC' and rel_2 == 'AM-LOC':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_loc = 1
            if coref_pred_tmp == 0 and rel_1 == 'AM-TMP' and rel_2 == 'AM-TMP':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_tmp = 1

    arg0_tensor = model.coref_role_embeds(torch.tensor(coref_pred_a0,
                                                       dtype=torch.long).to(device)).view(1,-1)
    arg1_tensor = model.coref_role_embeds(torch.tensor(coref_pred_a1,
                                                       dtype=torch.long).to(device)).view(1,-1)
    amloc_tensor = model.coref_role_embeds(torch.tensor(coref_pred_loc,
                                                        dtype=torch.long).to(device)).view(1,-1)
    amtmp_tensor = model.coref_role_embeds(torch.tensor(coref_pred_tmp,
                                                        dtype=torch.long).to(device)).view(1,-1)

    predicates_features_tensor = torch.cat([arg0_tensor, arg1_tensor, amloc_tensor, amtmp_tensor],1)

    return predicates_features_tensor


def float_to_tensor(float_num, device):
    '''
    Convert a floating point number to a tensor
    :param float_num: a floating point number
    :param device: Pytorch device (cpu/gpu)
    :return: a tensor
    '''
    float_tensor = torch.tensor([float(float_num)], requires_grad=False).to(device).view(1, -1)

    return float_tensor


def calc_q(cluster_1, cluster_2):
    '''
    Calculates the quality of merging two clusters, denotes by the proportion between
    the number gold coreferrential mention pairwise links (between the two clusters) and all the
    pairwise links.
    :param cluster_1: first cluster
    :param cluster_2: second cluster
    :return: the quality of merge (a number between 0 to 1)
    '''
    true_pairs = 0
    false_pairs = 0
    for mention_c1 in cluster_1.mentions.values():
        for mention_c2 in cluster_2.mentions.values():
            if mention_c1.gold_tag == mention_c2.gold_tag:
                true_pairs += 1
            else:
                false_pairs += 1

    return true_pairs/float(true_pairs + false_pairs)


def loadGloVe(glove_filename):
    '''
    Loads Glove word vectors.
    :param glove_filename: Glove file
    :return: vocab - list contains the vocabulary ,embd - list of word vectors
    '''
    vocab = []
    embd = []
    file = open(glove_filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if len(row) > 1:
            if row[0] != '':
                vocab.append(row[0])
                embd.append(row[1:])
                if len(row[1:]) != 300:
                    print(len(row[1:]))
    print('Loaded GloVe!')
    file.close()

    return vocab,embd


def get_sub_topics(doc_id):
    '''
    Extracts the sub-topic id from the document ID.
    :param doc_id: document id (string)
    :return: the sub-topic id (string)
    '''
    topic = doc_id.split('_')[0]
    if 'ecbplus' in doc_id:
        category = 'ecbplus'
    else:
        category = 'ecb'
    return '{}_{}'.format(topic, category)


def separate_clusters_to_sub_topics(clusters, is_event):
    '''
    Removes spurious cross sub-topics coreference link (used for experiments in Yang setup).
    :param clusters: a list of Cluster objects
    :param is_event: Clusters' type (event/entity)
    :return: new list of clusters, after spurious cross sub-topics coreference link were removed.
    '''
    new_clusters = []
    for cluster in clusters:
        sub_topics_to_clusters = {}
        for mention in cluster.mentions.values():
            mention_sub_topic = get_sub_topics(mention.doc_id)
            if mention_sub_topic not in sub_topics_to_clusters:
                sub_topics_to_clusters[mention_sub_topic] = []
            sub_topics_to_clusters[mention_sub_topic].append(mention)
        for sub_topic, mention_list in sub_topics_to_clusters.items():
            new_cluster = Cluster(is_event)
            for mention in mention_list:
                new_cluster.mentions[mention.mention_id] = mention
            new_clusters.append(new_cluster)

    return new_clusters


def set_coref_chain_to_mentions(clusters, is_event, is_gold, intersect_with_gold,):
    '''
    Sets the predicted cluster id to all mentions in the cluster
    :param clusters: predicted clusters (a list of Corpus objects)
    :param is_event: True, if clusters are event clusters, False otherwise - currently unused.
    :param is_gold: True, if the function sets gold mentions and false otherwise
     (it sets predicted mentions) - currently unused.
    :param intersect_with_gold: True, if the function sets predicted mentions that were matched
    with gold mentions (used in setting that requires to match predicted mentions with gold
    mentions - as in Yang's setting) , and false otherwise - currently unused.
    :param remove_singletons: True if the function ignores singleton clusters (as in Yang's setting)
    '''
    global clusters_count
    for cluster in clusters:
        cluster.cluster_id = clusters_count
        for mention in cluster.mentions.values():
            mention.cd_coref_chain = clusters_count
        clusters_count += 1


def save_check_point(model, fname):
    '''
    Saves Pytorch model to a file
    :param model: Pytorch model
    :param fname: output filename
    '''
    torch.save(model, fname)


def load_check_point(fname):
    '''
    Loads Pytorch model from a file
    :param fname: model's filename
    :return:Pytorch model
    '''
    return torch.load(fname)


def create_gold_clusters(mentions):
    '''
    Forms within document gold clusters.
    :param mentions: list of mentions
    :return: a dictionary contains the within document gold clusters (list)
    mapped by document id and the gold cluster ID.
    '''
    wd_clusters = {}
    for mention in mentions:
        mention_doc_id = mention.doc_id
        if mention_doc_id not in wd_clusters:
            wd_clusters[mention_doc_id] = {}
        mention_gold_tag = mention.gold_tag
        if mention_gold_tag not in wd_clusters[mention_doc_id]:
            wd_clusters[mention_doc_id][mention_gold_tag] = []
        wd_clusters[mention_doc_id][mention_gold_tag].append(mention)

    return wd_clusters


def create_gold_wd_clusters_organized_by_doc(mentions, is_event):
    '''
    Use the within document gold clusters (represented by lists of mention objects)
    to form Cluster objects.
    :param mentions: list of mentions
    :param is_event: Clusters' type (event/entity)
    :return: a dictionary contains gold within-document Cluster objects mapped by their document
    '''
    wd_clusters = create_gold_clusters(mentions)
    clusters_by_doc = {}

    for doc_id, gold_chain_in_doc in wd_clusters.items():
        for gold_chain_id, gold_chain in gold_chain_in_doc.items():
            cluster = Cluster(is_event)
            for mention in gold_chain:
                cluster.mentions[mention.mention_id] = mention
            if doc_id not in clusters_by_doc:
                clusters_by_doc[doc_id] = []
            clusters_by_doc[doc_id].append(cluster)

    return clusters_by_doc


def write_event_coref_results(corpus, out_dir, config_dict):
    '''
    Writes to a file (in a CoNLL format) the predicted event clusters (for evaluation).
    :param corpus: A Corpus object
    :param out_dir: output directory
    :param config_dict: configuration dictionary
    '''
    if not config_dict["test_use_gold_mentions"]:
        out_file = os.path.join(out_dir, 'CD_test_event_span_based.response_conll')
        write_span_based_cd_coref_clusters(corpus, out_file, is_event=True, is_gold=False,
                                           use_gold_mentions=config_dict["test_use_gold_mentions"])
    else:
        out_file = os.path.join(out_dir, 'CD_test_event_mention_based.response_conll')
        write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)

        out_file = os.path.join(out_dir, 'WD_test_event_mention_based.response_conll')
        write_mention_based_wd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)


def write_entity_coref_results(corpus, out_dir,config_dict):
    '''
    Writes to a file (in a CoNLL format) the predicted entity clusters (for evaluation).
    :param corpus: A Corpus object
    :param out_dir: output directory
    :param config_dict: configuration dictionary
    '''
    if not config_dict["test_use_gold_mentions"]:
        out_file = os.path.join(out_dir, 'CD_test_entity_span_based.response_conll')
        write_span_based_cd_coref_clusters(corpus, out_file, is_event=False, is_gold=False,
                                           use_gold_mentions=config_dict["test_use_gold_mentions"])
    else:
        out_file = os.path.join(out_dir, 'CD_test_entity_mention_based.response_conll')
        write_mention_based_cd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)

        out_file = os.path.join(out_dir, 'WD_test_entity_mention_based.response_conll')
        write_mention_based_wd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)


def create_event_cluster_bow_lexical_vec(event_cluster,model, device, use_char_embeds,
                                         requires_grad):
    '''
    Creates the semantically-dependent vector of a specific event cluster
    (average of mention's span vectors in the cluster)
    :param event_cluster: event cluster
    :param model: CDCorefScorer model
    :param device: Pytorch device (gpu/cpu)
    :param use_char_embeds: whether to use character embeddings
    :param requires_grad: whether the tensors require gradients (True for training time and
    False for inference time)
    :return: semantically-dependent vector of a specific event cluster
    (average of mention's span vectors in the cluster)
    '''
    if use_char_embeds:
        bow_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                              requires_grad=requires_grad).to(device).view(1, -1)
    else:
        bow_vec = torch.zeros(model.embedding_dim ,
                              requires_grad=requires_grad).to(device).view(1, -1)
    for event_mention in event_cluster.mentions.values():
        # creating lexical vector using the head word of each event mention in the cluster
        head = event_mention.mention_head
        head_tensor = find_word_embed(head, model, device)
        if use_char_embeds:
            char_tensor = get_char_embed(head, model, device)
            if not requires_grad:
                char_tensor = char_tensor.detach()
            cat_tensor = torch.cat([head_tensor, char_tensor], 1)
        else:
            cat_tensor = head_tensor
        bow_vec += cat_tensor

    return bow_vec / len(event_cluster.mentions.keys())


def create_entity_cluster_bow_lexical_vec(entity_cluster, model, device, use_char_embeds,
                                          requires_grad):
    '''
    Creates the semantically-dependent vector of a specific entity cluster
    (average of mention's span vectors in the cluster)
    :param entity_cluster: entity cluster
    :param model: CDCorefScorer model
    :param device: Pytorch device (gpu/cpu)
    :param use_char_embeds: whether to use character embeddings
    :param requires_grad: whether the tensors require gradients (True for training time and
    False for inference time)
    :return: semantically-dependent vector of a specific entity cluster
    (average of mention's span vectors in the cluster)
    '''
    if use_char_embeds:
        bow_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                              requires_grad=requires_grad).to(device).view(1, -1)
    else:
        bow_vec = torch.zeros(model.embedding_dim,
                              requires_grad=requires_grad).to(device).view(1, -1)
    for entity_mention in entity_cluster.mentions.values():
        # creating lexical vector using each entity mention in the cluster
        mention_bow = torch.zeros(model.embedding_dim,
                                  requires_grad=requires_grad).to(device).view(1, -1)
        mention_embeds = [find_word_embed(token, model, device)
                          for token in entity_mention.get_tokens()
                          if not is_stop(token)]
        if use_char_embeds:
            char_embeds = get_char_embed(entity_mention.mention_str, model, device)

        for word_tensor in mention_embeds:
            mention_bow += word_tensor

        mention_bow /= len(entity_mention.get_tokens())

        if use_char_embeds:
            if not requires_grad:
                char_embeds = char_embeds.detach()

            cat_tensor = torch.cat([mention_bow, char_embeds], 1)
        else:
            cat_tensor = mention_bow
        bow_vec += cat_tensor

    return bow_vec / len(entity_cluster.mentions.keys())


def find_mention_cluster_vec(mention_id, clusters):
    '''
    Fetches a semantically-dependent vector of a mention's cluster
    :param mention_id: mention ID (string)
    :param clusters: list of Cluster objects
    :return: semantically-dependent vector of a mention's cluster - Pytorch tensor with
     size (1, 350).
    '''
    for cluster in clusters:
        if mention_id in cluster.mentions:
            return cluster.lex_vec.detach()


def create_event_cluster_bow_arg_vec(event_cluster, entity_clusters, model, device):
    '''
    Creates the semantically-dependent vectors (of all roles) for all mentions
    in a specific event cluster.
    :param event_cluster: a Cluster object which contains EventMention objects.
    :param entity_clusters: current predicted entity clusters (a list)
    :param model: CDCorefScorer object
    :param device: Pytorch device
    '''
    for event_mention in event_cluster.mentions.values():
        event_mention.arg0_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        event_mention.arg1_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        event_mention.time_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        event_mention.loc_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        if event_mention.arg0 is not None:
            arg_vec = find_mention_cluster_vec(event_mention.arg0[1],entity_clusters)
            event_mention.arg0_vec = arg_vec.to(device)
        if event_mention.arg1 is not None:
            arg_vec = find_mention_cluster_vec(event_mention.arg1[1], entity_clusters)
            event_mention.arg1_vec = arg_vec.to(device)
        if event_mention.amtmp is not None:
            arg_vec = find_mention_cluster_vec(event_mention.amtmp[1], entity_clusters)
            event_mention.time_vec = arg_vec.to(device)
        if event_mention.amloc is not None:
            arg_vec = find_mention_cluster_vec(event_mention.amloc[1], entity_clusters)
            event_mention.loc_vec = arg_vec.to(device)


def create_entity_cluster_bow_predicate_vec(entity_cluster, event_clusters, model, device):
    '''
    Creates the semantically-dependent vectors (of all roles) for all mentions
    in a specific event cluster.
    :param entity_cluster: a Cluster object which contains EntityMention objects.
    :param event_clusters: current predicted entity clusters (a list)
    :param model: CDCorefScorer object
    :param device: Pytorch device
    '''
    for entity_mention in entity_cluster.mentions.values():
        entity_mention.arg0_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        entity_mention.arg1_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        entity_mention.time_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        entity_mention.loc_vec = torch.zeros(model.embedding_dim + model.char_hidden_dim,
                                  requires_grad=False).to(device).view(1, -1)
        predicates_dict = entity_mention.predicates
        for predicate_id, rel in predicates_dict.items():
            if rel == 'A0':
                pred_vec = find_mention_cluster_vec(predicate_id[1], event_clusters)
                entity_mention.arg0_vec = pred_vec.to(device)
            elif rel == 'A1':
                pred_vec = find_mention_cluster_vec(predicate_id[1], event_clusters)
                entity_mention.arg1_vec = pred_vec.to(device)
            elif rel == 'AM-TMP':
                pred_vec = find_mention_cluster_vec(predicate_id[1], event_clusters)
                entity_mention.time_vec = pred_vec.to(device)
            elif rel == 'AM-LOC':
                pred_vec = find_mention_cluster_vec(predicate_id[1], event_clusters)
                entity_mention.loc_vec = pred_vec.to(device)


def update_lexical_vectors(clusters, model, device ,is_event, requires_grad):
    '''
    Updates for each cluster its average vector of all mentions' span representations
    (Used to form the semantically-dependent vectors)
    :param clusters: list of Cluster objects (event/entity clsuters)
    :param model: CDCorefScorer object, should be an event model if clusters are event clusters
    (and the same with entities)
    :param device: Pytorch device
    :param is_event: True if clusters are event clusters and false otherwise (clusters are entity
    clusters)
    :param requires_grad: True if tensors require gradients (for training time) , and
    False for inference time.
    '''
    for cluster in clusters:
        if is_event:
            lex_vec = create_event_cluster_bow_lexical_vec(cluster, model, device,
                                                           use_char_embeds=True,
                                                           requires_grad=requires_grad)
        else:
            lex_vec = create_entity_cluster_bow_lexical_vec(cluster, model, device,
                                                            use_char_embeds=True,
                                                            requires_grad=requires_grad)

        cluster.lex_vec = lex_vec


def update_args_feature_vectors(clusters, other_clusters ,model ,device, is_event):
    '''
     Updates for each mention in clusters its semantically-dependent vectors
    :param clusters: current event/entity clusters (list of Cluster objects)
    :param other_clusters: should be the current event clusters if clusters = entity clusters
    and vice versa.
    :param model: event/entity model (should be according to clusters parameter)
    :param device: Pytorch device
    :param is_event: True if clusters are event clusters and False if they are entity clusters.
    '''
    for cluster in clusters:
        if is_event:
            # Use an average of span representations to represent arguments/predicates clusters.
            create_event_cluster_bow_arg_vec(cluster, other_clusters, model, device)
        else:
            create_entity_cluster_bow_predicate_vec(cluster, other_clusters, model, device)


def generate_cluster_pairs(clusters, is_train):
    '''
    Given list of clusters, this function generates candidate cluster pairs (for training/inference).
    The function under-samples cluster pairs without any coreference links
     when generating cluster pairs for training and the current number of clusters in the
    current topic is larger than 300.
    :param clusters: current clusters
    :param is_train: True if the function generates candidate cluster pairs for training time
    and False, for inference time (without under-sampling)
    :return: pairs - generated cluster pairs (potentially with under-sampling)
    , test_pairs -  all cluster pairs
    '''

    positive_pairs_count = 0
    negative_pairs_count = 0
    pairs = []
    test_pairs = []

    use_under_sampling = True if (len(clusters) > 300 and is_train) else False

    if len(clusters) < 500:
        p = 0.7
    else:
        p = 0.6

    print('Generating cluster pairs...')
    logging.info('Generating cluster pairs...')

    print('Initial number of clusters = {}'.format(len(clusters)))
    logging.info('Initial number of clusters = {}'.format(len(clusters)))

    if use_under_sampling:
        print('Using under sampling with p = {}'.format(p))
        logging.info('Using under sampling with p = {}'.format(p))

    for cluster_1 in clusters:
        for cluster_2 in clusters:
            if cluster_1 != cluster_2:
                if is_train:
                    q = calc_q(cluster_1, cluster_2)
                    if (cluster_1, cluster_2, q) \
                            not in pairs and (cluster_2, cluster_1, q) not in pairs:
                        add_to_training = False if use_under_sampling else True
                        if q > 0:
                            add_to_training = True
                            positive_pairs_count += 1
                        if q == 0 and random.random() < p:
                            add_to_training = True
                            negative_pairs_count += 1
                        if add_to_training:
                            pairs.append((cluster_1, cluster_2, q))
                        test_pairs.append((cluster_1, cluster_2))
                else:
                    if (cluster_1, cluster_2) not in pairs and \
                            (cluster_2, cluster_1) not in pairs:
                        pairs.append((cluster_1, cluster_2))

    print('Number of generated cluster pairs = {}'.format(len(pairs)))
    logging.info('Number of generated cluster pairs = {}'.format(len(pairs)))

    return pairs, test_pairs


def get_mention_span_rep(mention, device, model, docs, is_event, requires_grad):
    '''
    Creates for a mention its context and span text vectors and concatenates them.
    :param mention: an Mention object (either an EventMention or an EntityMention)
    :param device: Pytorch device
    :param model: CDCorefScorer object, should be in the same type as the mention
    :param docs: the current topic's documents
    :param is_event: True if mention is an event mention and False if it is an entity mention
    :param requires_grad: True if tensors require gradients (for training time) , and
    False for inference time.
    :return: a tensor with size (1, 1374)
    '''

    span_tensor = mention.head_elmo_embeddings.to(device).view(1,-1)

    if is_event:
        head = mention.mention_head
        head_tensor = find_word_embed(head, model, device)
        char_embeds = get_char_embed(head, model, device)
        mention_tensor = torch.cat([head_tensor, char_embeds], 1)
    else:
        mention_bow = torch.zeros(model.embedding_dim, requires_grad=requires_grad).to(device).view(1, -1)
        mention_embeds = [find_word_embed(token, model, device) for token in mention.get_tokens()
                          if not is_stop(token)]

        for mention_word_tensor in mention_embeds:
            mention_bow += mention_word_tensor
        char_embeds = get_char_embed(mention.mention_str, model, device)

        if len(mention_embeds) > 0:
            mention_bow = mention_bow / float(len(mention_embeds))

        mention_tensor = torch.cat([mention_bow, char_embeds], 1)

    mention_span_rep = torch.cat([span_tensor, mention_tensor], 1)

    if requires_grad:
        if not mention_span_rep.requires_grad:
            logging.info('mention_span_rep does not require grad ! (warning)')
    else:
        mention_span_rep = mention_span_rep.detach()

    return mention_span_rep


def create_mention_span_representations(mentions, model, device, topic_docs, is_event,
                                        requires_grad):
    '''
    Creates for a set of mentions their context and span text vectors.
    :param mentions: a list of Mention objects (an EventMention or an EntityMention)
    :param model: CDCorefScorer object, should be in the same type as the mentions
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param is_event: True if mention is an event mention and False if it is an entity mention
    :param requires_grad: True if tensors require gradients (for training time) , and
    False for inference time.
     embeddings (performs worse than ELMo embeddings)
    '''
    for mention in mentions:
        mention.span_rep = get_mention_span_rep(mention, device, model, topic_docs,
                                                is_event, requires_grad)


def mention_pair_to_model_input(pair, model, device, topic_docs, is_event, requires_grad,
                                 use_args_feats, use_binary_feats, other_clusters):
    '''
    Given a pair of mentions, the function builds the input to the network (the mention-pair
    representation) and returns it.
    :param pair: a tuple of two Mention objects (should be of the same type - events or entities)
    :param model: CDCorefScorer object (should be in the same type as pair - event or entity Model)
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param is_event: True if pair is an event mention pair and False if it's
     an entity mention pair.
    :param requires_grad: True if tensors require gradients (for training time) , and
    False for inference time.
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    them.
    :param other_clusters: should be the current event clusters if pair is an entity mention pair
    and vice versa.
    :return: the mention-pair representation - a tensor of size (1,X), when X = 8522 in the full
    joint model (without any ablation)
    '''
    mention_1 = pair[0]
    mention_2 = pair[1]

    # create span representation
    if requires_grad :
        mention_1.span_rep = get_mention_span_rep(mention_1, device, model, topic_docs,
                                                  is_event, requires_grad)
        mention_2.span_rep = get_mention_span_rep(mention_2, device, model, topic_docs,
                                                  is_event, requires_grad)
    span_rep_1 = mention_1.span_rep
    span_rep_2 = mention_2.span_rep

    if use_args_feats:
        mention_1_tensor = torch.cat([span_rep_1, mention_1.arg0_vec, mention_1.arg1_vec,
                                      mention_1.loc_vec, mention_1.time_vec], 1)
        mention_2_tensor = torch.cat([span_rep_2, mention_2.arg0_vec,mention_2.arg1_vec,
                                      mention_2.loc_vec,mention_2.time_vec], 1)

    else:
        mention_1_tensor = span_rep_1
        mention_2_tensor = span_rep_2

    if model.use_mult and model.use_diff:
        mention_pair_tensor = torch.cat([mention_1_tensor, mention_2_tensor,
                                         mention_1_tensor - mention_2_tensor,
                                         mention_1_tensor * mention_2_tensor], 1)
    elif model.use_mult:
        mention_pair_tensor = torch.cat([mention_1_tensor, mention_2_tensor,
                                         mention_1_tensor * mention_2_tensor], 1)
    elif model.use_diff:
        mention_pair_tensor = torch.cat([mention_1_tensor, mention_2_tensor,
                                         mention_1_tensor - mention_2_tensor], 1)

    if use_binary_feats:
        if is_event:
            binary_feats = create_args_features_vec(mention_1, mention_2, other_clusters,
                                                    device, model)
        else:
            binary_feats = create_predicates_features_vec(mention_1, mention_2, other_clusters,
                                                          device, model)

        mention_pair_tensor = torch.cat([mention_pair_tensor,binary_feats], 1)

    mention_pair_tensor = mention_pair_tensor.to(device)

    return mention_pair_tensor


def train_pairs_batch_to_model_input(batch_pairs, model, device, topic_docs, is_event,
                                      use_args_feats, use_binary_feats, other_clusters):
    '''
    Creates input tensors (mention pair representations) to all mention pairs in the batch
    (for training time).
    :param batch_pairs: a list of mention pairs (in the size of the batch)
    :param model: CDCorefScorer object (should be in the same type as batch_pairs
     - event or entity Model)
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param is_event:  True if pairs are event mention pairs and False if they are
    entity mention pairs.
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    them.
    :param other_clusters: should be the current event clusters if batch_pairs are entity mention
     pairs and vice versa.
    :return: batch_pairs_tensor - a tensor of the mention pair representations
    according to the batch size, q_pairs_tensor - a tensor of the pairs' gold labels
    '''
    tensors_list = []
    q_list = []
    for pair in batch_pairs:
        mention_pair_tensor = mention_pair_to_model_input(pair, model, device, topic_docs,
                                                          is_event, requires_grad=True,
                                                          use_args_feats=use_args_feats,
                                                          use_binary_feats=use_binary_feats,
                                                          other_clusters=other_clusters)
        if not mention_pair_tensor.requires_grad:
            logging.info('mention_pair_tensor does not require grad ! (warning)')

        tensors_list.append(mention_pair_tensor)

        q = 1.0 if pair[0].gold_tag == pair[1].gold_tag else 0.0
        q_tensor = float_to_tensor(q, device)
        q_list.append(q_tensor)

    batch_pairs_tensor = torch.cat(tensors_list, 0)
    q_pairs_tensor = torch.cat(q_list, 0)

    return batch_pairs_tensor, q_pairs_tensor


def train(cluster_pairs, model, optimizer, loss_function, device, topic_docs, epoch,
          topics_counter, topics_num, config_dict, is_event, other_clusters):
    '''
    Trains a model using a given set of cluster pairs, a specific optimizer and a loss function.
    The model is trained on all mention pairs between each cluster pair.
    :param cluster_pairs: list of clusters pairs
    :param model: CDCorefModel object
    :param optimizer: Pytorch optimizer
    :param loss_function: Pytorch loss function
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param epoch: current epoch
    :param topics_counter: current topic number
    :param topics_num: total number of topics
    :param config_dict: configuration dictionary, stores the configuration of the experiment
    :param is_event: True, if model is an event model and False if it's an entity model
    :param other_clusters: should be the current event clusters if the function trains
     an entity model and vice versa.
    '''
    batch_size = config_dict["batch_size"]
    mode = 'Event' if is_event else 'Entity'
    retain_graph = False
    epochs = config_dict["regressor_epochs"]
    random.shuffle(cluster_pairs)

    # creates mention pairs and their true labels (creates max 100,000 mention pairs - due to memory constrains)
    pairs = cluster_pairs_to_mention_pairs(cluster_pairs)
    random.shuffle(pairs)

    for reg_epoch in range(0, epochs):
        samples_count = 0
        batches_count = 0
        total_loss = 0
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size) if i + batch_size < len(pairs)]
        for batch_pairs in batches:
            samples_count += batch_size
            batches_count += 1
            batch_tensor, q_tensor = train_pairs_batch_to_model_input(batch_pairs, model,
                                                                device, topic_docs, is_event,
                                                                      config_dict["use_args_feats"],
                                                                      config_dict["use_binary_feats"],
                                                                      other_clusters)

            model.zero_grad()
            output = model(batch_tensor)
            loss = loss_function(output, q_tensor)
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            total_loss += loss.item()

            if samples_count % config_dict["log_interval"] == 0:
                print('epoch {}, topic {}/{} - {} model '
                      ' [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                    epoch,topics_counter, topics_num, mode, samples_count, len(pairs),
                    100. * samples_count / len(pairs), (total_loss/float(batches_count))))

            del batch_tensor, q_tensor


def cluster_pair_to_mention_pair(pair):
    '''
    Given a cluster pair, the function extracts all the mention pairs between the two clusters
    :param pair: a cluster pair (tuple of two Cluster objects)
    :return: a list contains tuples of Mention object pairs (EventMention/EntityMention)
    '''
    mention_pairs = []
    cluster_1 = pair[0]
    cluster_2 = pair[1]

    c1_mentions = cluster_1.mentions.values()
    c2_mentions = cluster_2.mentions.values()

    for mention_1 in c1_mentions:
        for mention_2 in c2_mentions:
            mention_pairs.append((mention_1, mention_2))

    return mention_pairs


def cluster_pairs_to_mention_pairs(cluster_pairs):
    '''
    Generates all mention pairs between all cluster pairs
    :param cluster_pairs: cluster pairs (tuples of two Cluster objects)
    :return: a list contains tuples of Mention object pairs (EventMention/EntityMention)
    '''
    th = 100000
    mention_pairs = []

    for pair in cluster_pairs:

        mention_pairs.extend(cluster_pair_to_mention_pair(pair))

        if len(mention_pairs) > th: # up to 100,000 pairs (due to memory constrains)
            break

    return mention_pairs


def test_pairs_batch_to_model_input(batch_pairs, model, device, topic_docs, is_event,
                                     use_args_feats, use_binary_feats, other_clusters):

    '''
    Creates input tensors (mention pair representations) for all mention pairs in the batch
    (for inference time).
    :param batch_pairs: a list of mention pairs (in the size of the batch)
    :param model: CDCorefScorer object (should be in the same type as batch_pairs
     - event or entity Model)
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param is_event:  True if pairs are event mention pairs and False if they are
    entity mention pairs.
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    them.
    :param other_clusters: should be the current event clusters if batch_pairs are entity mention
     pairs and vice versa.
    :return: batch_pairs_tensor - a tensor of the mention pair representations
    according to the batch size, q_pairs_tensor - a tensor of the pairs' gold labels

    '''
    tensors_list = []
    for pair in batch_pairs:
        mention_pair_tensor = mention_pair_to_model_input(pair, model, device, topic_docs,
                                                          is_event, requires_grad=False,
                                                          use_args_feats=use_args_feats,
                                                          use_binary_feats=use_binary_feats,
                                                          other_clusters=other_clusters)
        tensors_list.append(mention_pair_tensor)

    batch_pairs_tensor = torch.cat(tensors_list, 0)

    return batch_pairs_tensor


def get_batches(mention_pairs, batch_size):
    '''
    Splits the mention pairs to batches (specifically this function used during inference time)
    :param mention_pairs: a list contains a tuples of mention pairs
    :param batch_size: the batch size (integer)
    :return: list of lists, when each inner list contains each batch's pairs
    '''
    batches = [mention_pairs[i:i + batch_size] for i in
               range(0, len(mention_pairs),batch_size) if i + batch_size < len(mention_pairs)]
    diff = len(mention_pairs) - len(batches)*batch_size
    if diff > 0:
        batches.append(mention_pairs[-diff:])

    return batches

def key_with_max_val(d):
    """ a) creates a list of the dict's keys and values;
        b) returns the key with the max value and the max value"""
    v = list(d.values()) #scores
    k = list(d.keys()) #pairs

    np_scores = np.asarray(v)
    best_ix = np.argmax(np_scores)
    best_score = np_scores[best_ix]

    return k[v.index(best_score)], best_score


def merge_clusters(pair_to_merge, clusters ,other_clusters, is_event,
                   model, device, topic_docs, curr_pairs_dict,
                   use_args_feats, use_binary_feats):
    '''
    This function:
    1. Merges a cluster pair and update its span vector (average of all its mentions' span
    representations)
    2. Removes the merged pair from the clusters list and the scores dict
    3. Removes from the pair-score dict all the pairs that contain one of the merged clusters
    4. Creates new cluster pairs
    5. Adds the new pairs to the pair-score dict and lets the model to assign score to each pair
    :param pair_to_merge: a tuple of two Cluster objects that were chosen to get merged.
    :param clusters: current event/entity clusters (of the same type of pair_to_merge)
    :param other_clusters: should be the current event clusters if clusters are entity clusters
     and vice versa.
    :param is_event: True if pair_to_merge is an event pair  and False if they it's an
    entity pair.
    :param model: CDCorefModel object
    :param device: Pytorch device object
    :param topic_docs: current topic's documents
    :param curr_pairs_dict: dictionary contains the current candidate cluster pairs
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    them.
    '''
    cluster_i = pair_to_merge[0]
    cluster_j = pair_to_merge[1]
    new_cluster = Cluster(is_event)
    new_cluster.mentions.update(cluster_j.mentions)
    new_cluster.mentions.update(cluster_i.mentions)

    keys_pairs_dict = list(curr_pairs_dict.keys())
    for pair in keys_pairs_dict:
        cluster_pair = (pair[0], pair[1])
        if cluster_i in cluster_pair or cluster_j in cluster_pair:
            del curr_pairs_dict[pair]

    clusters.remove(cluster_i)
    clusters.remove(cluster_j)
    clusters.append(new_cluster)

    if is_event:
        lex_vec = create_event_cluster_bow_lexical_vec(new_cluster, model, device,
                                                       use_char_embeds=True,
                                                       requires_grad=False)
    else:
        lex_vec = create_entity_cluster_bow_lexical_vec(new_cluster, model, device,
                                                        use_char_embeds=True,
                                                        requires_grad=False)

    new_cluster.lex_vec = lex_vec

    # create arguments features for the new cluster
    update_args_feature_vectors([new_cluster], other_clusters, model, device, is_event)

    new_pairs = []
    for cluster in clusters:
        if cluster != new_cluster:
            new_pairs.append((cluster, new_cluster))

    # create scores for the new pairs
    for pair in new_pairs:
        pair_score = assign_score(pair, model, device, topic_docs, is_event,
                                  use_args_feats, use_binary_feats, other_clusters)
        curr_pairs_dict[pair] = pair_score


def assign_score(cluster_pair, model, device, topic_docs, is_event, use_args_feats,
                 use_binary_feats, other_clusters):
    '''
    Assigns coreference (or quality of merge) score to a cluster pair by averaging the mention-pair
    scores predicted by the model.
    :param cluster_pair: a tuple of two Cluster objects
    :param model: CDCorefScorer object
    :param device: Pytorch device
    :param topic_docs: current topic's documents
    :param is_event: True if cluster_pair is an event pair and False if it's an entity pair
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    them.
    :param other_clusters: should be the current event clusters if cluster_pair is an entity pair
     and vice versa.
    :return: The average mention pairwise score
    '''
    mention_pairs = cluster_pair_to_mention_pair(cluster_pair)
    batches = get_batches(mention_pairs, 256)
    pairs_count = 0
    scores_sum = 0
    for batch_pairs in batches:
        batch_tensor = test_pairs_batch_to_model_input(batch_pairs, model, device,
                                                       topic_docs, is_event,
                                                       use_args_feats=use_args_feats,
                                                       use_binary_feats=use_binary_feats,
                                                       other_clusters=other_clusters)

        model_scores = model(batch_tensor).detach().cpu().numpy()
        scores_sum += float(np.sum(model_scores))
        pairs_count += len(model_scores)

        del batch_tensor

    return scores_sum/float(pairs_count)


def merge(clusters, cluster_pairs, other_clusters,model, device, topic_docs, epoch, topics_counter,
          topics_num, threshold, is_event, use_args_feats, use_binary_feats):
    '''
    Merges cluster pairs in agglomerative manner till it reaches a pre-defined threshold. In each step, the function merges
    cluster pair with the highest score, and updates the candidate cluster pairs according to the
    current merge.
    Note that all Cluster objects in clusters should have the same type (event or entity but
    not both of them).
    other_clusters are fixed during merges and should have the opposite type
    i.e. if clusters are event clusters, so other_clusters will be the entity clusters.

    :param clusters: a list of Cluster objects of the same type (event/entity)
    :param cluster_pairs: a list of the cluster pairs (tuples)
    :param other_clusters: a list of Cluster objects with the opposite type to clusters.
    Stays fixed during merging operations on clusters.
    :param model: CDCorefScorer object with the same type as clusters.
    :param device: Pytorch device
    :param topic_docs: current topic's documents
    :param epoch: current epoch (relevant to training)
    :param topics_counter: current topic number
    :param topics_num: total number of topics
    :param threshold: merging threshold
    :param is_event: True if clusters are event clusters and false if they are entity clusters
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    '''
    print('Initialize cluster pairs scores... ')
    logging.info('Initialize cluster pairs scores... ')
    # initializes the pairs-scores dict
    pairs_dict = {}
    mode = 'event' if is_event else 'entity'
    # init the scores (that the model assigns to the pairs)
    for pair in cluster_pairs:
        pair_score = assign_score(pair, model, device, topic_docs, is_event,
                                  use_args_feats,use_binary_feats, other_clusters)
        pairs_dict[pair] = pair_score

    while True:
        # finds max pair (break if we can't find one  - max score < threshold)
        if len(pairs_dict) < 2:
            print('Less the 2 clusters had left, stop merging!')
            logging.info('Less the 2 clusters had left, stop merging!')
            break
        max_pair, max_score = key_with_max_val(pairs_dict)

        if max_score > threshold:
            print('epoch {} topic {}/{} - merge {} clusters with score {} clusters : {} {}'.format(
                epoch, topics_counter, topics_num, mode, str(max_score), str(max_pair[0]),
                str(max_pair[1])))
            logging.info('epoch {} topic {}/{} - merge {} clusters with score {} clusters : {} {}'.format(
                epoch, topics_counter, topics_num, mode, str(max_score), str(max_pair[0]),
                str(max_pair[1])))
            merge_clusters(max_pair, clusters, other_clusters, is_event,
                           model, device, topic_docs, pairs_dict, use_args_feats,
                           use_binary_feats)
        else:
            print('Max score = {} is lower than threshold = {},'
                  ' stopped merging!'.format(max_score, threshold))
            logging.info('Max score = {} is lower than threshold = {},' \
                         ' stopped merging!'.format(max_score, threshold))
            break


def test_model(clusters, other_clusters, model, device, topic_docs, is_event, epoch,
               topics_counter, topics_num, threshold, use_args_feats,
               use_binary_feats):
    '''
    Runs the inference procedure for a specific model (event/entity model).
    :param clusters: a list of Cluster objects of the same type (event/entity)
    :param other_clusters: a list of Cluster objects with the opposite type to clusters.
    Stays fixed during merging operations on clusters.
    :param model: CDCorefScorer object with the same type as clusters.
    :param device: Pytorch device
    :param topic_docs: current topic's documents
    :param epoch: current epoch (relevant to training)
    :param topics_counter: current topic number
    :param topics_num: total number of topics
    :param threshold: merging threshold
    :param is_event: True if clusters are event clusters and false if they are entity clusters
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate them.
    '''

    # updating the semantically - dependent vectors according to other_clusters
    update_args_feature_vectors(clusters, other_clusters, model, device, is_event)

    # generating candidate cluster pairs
    cluster_pairs, _ = generate_cluster_pairs(clusters, is_train=False)

    # merging clusters pairs till reaching a pre-defined threshold
    merge(clusters, cluster_pairs, other_clusters,model, device, topic_docs, epoch,
          topics_counter, topics_num, threshold, is_event, use_args_feats,
          use_binary_feats)


def test_models(test_set, cd_event_model,cd_entity_model, device,
                config_dict, write_clusters, out_dir, doc_to_entity_mentions, analyze_scores):
    '''
    Runs the inference procedure for both event and entity models calculates the B-cubed
    score of their predictions.
    :param test_set: Corpus object containing the test documents.
    :param cd_event_model: CDCorefScorer object models cross-document event coreference decisions
    :param cd_entity_model: CDCorefScorer object models cross-document entity coreference decisions
    :param device: Pytorch device
    :param config_dict: a dictionary contains the experiment's configurations
    :param write_clusters: whether to write predicted clusters to file (for analysis purpose)
    :param out_dir: output files directory
    :param doc_to_entity_mentions: a dictionary contains the within-document (WD) entity coreference
    chains predicted by an external (WD) entity coreference system.
    :param analyze_scores: whether to save representations and Corpus objects for analysis
    :return: B-cubed scores for the predicted event and entity clusters
    '''

    global clusters_count
    clusters_count = 1
    event_errors = []
    entity_errors = []
    all_event_clusters = []
    all_entity_clusters = []

    if config_dict["load_predicted_topics"]:
        topics = load_predicted_topics(test_set,config_dict) # use the predicted sub-topics
    else:
        topics = test_set.topics # use the gold sub-topics

    topics_num = len(topics.keys())
    topics_counter = 0
    topics_keys = topics.keys()
    epoch = 0 #
    all_event_mentions = []
    all_entity_mentions = []

    with torch.no_grad():
        for topic_id in topics_keys:
            topic = topics[topic_id]
            topics_counter += 1

            logging.info('=========================================================================')
            logging.info('Topic {}:'.format(topic_id))
            print('Topic {}:'.format(topic_id))

            event_mentions, entity_mentions = topic_to_mention_list(topic,
                                                                    is_gold=config_dict["test_use_gold_mentions"])

            all_event_mentions.extend(event_mentions)
            all_entity_mentions.extend(entity_mentions)

            # create span rep for both entity and event mentions
            create_mention_span_representations(event_mentions, cd_event_model, device,
                                                topic.docs, is_event=True,
                                                requires_grad=False)
            create_mention_span_representations(entity_mentions, cd_entity_model, device,
                                                topic.docs, is_event=False,
                                                requires_grad=False)

            print('number of event mentions : {}'.format(len(event_mentions)))
            print('number of entity mentions : {}'.format(len(entity_mentions)))
            logging.info('number of event mentions : {}'.format(len(event_mentions)))
            logging.info('number of entity mentions : {}'.format(len(entity_mentions)))
            topic.event_mentions = event_mentions
            topic.entity_mentions = entity_mentions

            # initialize within-document entity clusters with the output of within-document system
            wd_entity_clusters = init_entity_wd_clusters(entity_mentions, doc_to_entity_mentions)

            topic_entity_clusters = []
            for doc_id, clusters in wd_entity_clusters.items():
                topic_entity_clusters.extend(clusters)

            # initialize event clusters as singletons
            topic_event_clusters = init_cd(event_mentions, is_event=True)

            # init cluster representation
            update_lexical_vectors(topic_entity_clusters, cd_entity_model, device,
                                   is_event=False, requires_grad=False)
            update_lexical_vectors(topic_event_clusters, cd_event_model, device,
                                   is_event=True, requires_grad=False)

            entity_th = config_dict["entity_merge_threshold"]
            event_th = config_dict["event_merge_threshold"]

            for i in range(1,config_dict["merge_iters"]+1):
                print('Iteration number {}'.format(i))
                logging.info('Iteration number {}'.format(i))

                # Merge entities
                print('Merge entity clusters...')
                logging.info('Merge entity clusters...')
                test_model(clusters=topic_entity_clusters, other_clusters=topic_event_clusters,
                           model=cd_entity_model, device=device, topic_docs=topic.docs,is_event=False,epoch=epoch,
                           topics_counter=topics_counter, topics_num=topics_num,
                           threshold=entity_th,
                           use_args_feats=config_dict["use_args_feats"],
                           use_binary_feats=config_dict["use_binary_feats"])
                # Merge events
                print('Merge event clusters...')
                logging.info('Merge event clusters...')
                test_model(clusters=topic_event_clusters, other_clusters=topic_entity_clusters,
                           model=cd_event_model,device=device, topic_docs=topic.docs, is_event=True,epoch=epoch,
                           topics_counter=topics_counter, topics_num=topics_num,
                           threshold=event_th,
                           use_args_feats=config_dict["use_args_feats"],
                           use_binary_feats=config_dict["use_binary_feats"])

            set_coref_chain_to_mentions(topic_event_clusters, is_event=True,
                                        is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
            set_coref_chain_to_mentions(topic_entity_clusters, is_event=False,
                                        is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)

            if write_clusters:
                # Save for analysis
                all_event_clusters.extend(topic_event_clusters)
                all_entity_clusters.extend(topic_entity_clusters)

                with open(os.path.join(out_dir, 'entity_clusters.txt'), 'a') as entity_file_obj:
                    write_clusters_to_file(topic_entity_clusters, entity_file_obj, topic_id)
                    entity_errors.extend(collect_errors(topic_entity_clusters, topic_event_clusters, topic.docs,
                                                        is_event=False))

                with open(os.path.join(out_dir, 'event_clusters.txt'), 'a') as event_file_obj:
                    write_clusters_to_file(topic_event_clusters, event_file_obj, topic_id)
                    event_errors.extend(collect_errors(topic_event_clusters, topic_entity_clusters, topic.docs,
                                                       is_event=True))

        if write_clusters:
            write_event_coref_results(test_set, out_dir, config_dict)
            write_entity_coref_results(test_set, out_dir, config_dict)
            sample_errors(event_errors, os.path.join(out_dir,'event_errors'))
            sample_errors(entity_errors, os.path.join(out_dir,'entity_errors'))

    if analyze_scores:
        # Save mention representations
        save_mention_representations(all_event_clusters, out_dir, is_event=True)
        save_mention_representations(all_entity_clusters, out_dir, is_event=False)

        # Save topics for analysis
        with open(os.path.join(out_dir,'test_topics'), 'wb') as f:
            cPickle.dump(topics, f)

    if config_dict["test_use_gold_mentions"]:
        event_predicted_lst = [event.cd_coref_chain for event in all_event_mentions]
        true_labels = [event.gold_tag for event in all_event_mentions]
        true_clusters_set = set(true_labels)

        labels_mapping = {}
        for label in true_clusters_set:
            labels_mapping[label] = len(labels_mapping)

        event_gold_lst = [labels_mapping[label] for label in true_labels]
        event_r, event_p, event_b3_f1 = bcubed(event_gold_lst, event_predicted_lst)

        entity_predicted_lst = [entity.cd_coref_chain for entity in all_entity_mentions]
        true_labels = [entity.gold_tag for entity in all_entity_mentions]
        true_clusters_set = set(true_labels)

        labels_mapping = {}
        for label in true_clusters_set:
            labels_mapping[label] = len(labels_mapping)

        entity_gold_lst = [labels_mapping[label] for label in true_labels]
        entity_r, entity_p, entity_b3_f1 = bcubed(entity_gold_lst, entity_predicted_lst)

        return event_b3_f1, entity_b3_f1

    else:
        print('Using predicted mentions, can not calculate CoNLL F1')
        logging.info('Using predicted mentions, can not calculate CoNLL F1')
        return 0,0


def init_clusters_with_lemma_baseline(mentions, is_event):
    '''
    Initializes clusters for agglomerative clustering with the output of the head lemma baseline
    (used for experiments)
    :param mentions: list of Mention objects (EventMention/EntityMention objects)
    :param is_event: True if mentions are event mentions and False if they are entity mentions
    :return: list of Cluster objects
    '''
    mentions_by_head_lemma = {}
    clusters = []

    for mention in mentions:
        if mention.mention_head_lemma not in mentions_by_head_lemma:
            mentions_by_head_lemma[mention.mention_head_lemma] = []
        mentions_by_head_lemma[mention.mention_head_lemma].append(mention)

    for head_lemma, mentions in mentions_by_head_lemma.items():
        cluster = Cluster(is_event=is_event)
        for mention in mentions:
            cluster.mentions[mention.mention_id] = mention
        clusters.append(cluster)

    return clusters


def mention_data_to_string(mention, other_clusters, is_event,topic_docs):
    '''
    Creates a string representing a mention's data
    :param mention: a Mention object (EventMention/EntityMention)
    :param other_clusters: current entity clusters if mention is an event mention and vice versa
    :param is_event: True if mention is an event mention and False if it's an entity mention.
    :param topic_docs: current topic's documents
    :return: a string representing a mention's data
    '''
    strings = ['mention: {}_{}'.format(mention.mention_str,mention.gold_tag)]
    if is_event:
        if mention.arg0 is not None:
            arg0_cluster = find_mention_cluster(mention.arg0[1], other_clusters)
            gold_arg0_chain = arg0_cluster.mentions[mention.arg0[1]].gold_tag
            strings.append('arg0: {}_{}_{}'.format(mention.arg0[0], arg0_cluster.cluster_id,
                                                   gold_arg0_chain))
        if mention.arg1 is not None:
            arg1_cluster = find_mention_cluster(mention.arg1[1], other_clusters)
            gold_arg1_chain = arg1_cluster.mentions[mention.arg1[1]].gold_tag
            strings.append('arg1: {}_{}_{}'.format(mention.arg1[0], arg1_cluster.cluster_id,
                                                   gold_arg1_chain))
        if mention.amtmp is not None:
            amtmp_cluster = find_mention_cluster(mention.amtmp[1], other_clusters)
            gold_amtmp_chain = amtmp_cluster.mentions[mention.amtmp[1]].gold_tag
            strings.append('amtmp: {}_{}_{}'.format(mention.amtmp[0], amtmp_cluster.cluster_id,
                                                    gold_amtmp_chain))
        if mention.amloc is not None:
            amloc_cluster = find_mention_cluster(mention.amloc[1], other_clusters)
            gold_amloc_chain = amloc_cluster.mentions[mention.amloc[1]].gold_tag
            strings.append('amloc: {}_{}_{}'.format(mention.amloc[0], amloc_cluster.cluster_id,
                                                    gold_amloc_chain))
    else:
        for pred, rel in mention.predicates.items():
            if rel == 'A0':
                arg0_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_arg0_chain = arg0_cluster.mentions[pred[1]].gold_tag
                strings.append('arg0_p: {}_{}_{}'.format(pred[0], arg0_cluster.cluster_id,
                                                         gold_arg0_chain))
            elif rel == 'A1':
                arg1_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_arg1_chain = arg1_cluster.mentions[pred[1]].gold_tag
                strings.append('arg1_p: {}_{}_{}'.format(pred[0], arg1_cluster.cluster_id,
                                                         gold_arg1_chain))
            elif rel == 'AM-TMP':
                amtmp_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_amtmp_chain = amtmp_cluster.mentions[pred[1]].gold_tag
                strings.append('amtmp_p: {}_{}_{}'.format(pred[0], amtmp_cluster.cluster_id,
                                                          gold_amtmp_chain))
            elif rel == 'AM-LOC':
                amloc_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_amloc_chain = amloc_cluster.mentions[pred[1]].gold_tag
                strings.append('amloc_p: {}_{}_{}'.format(pred[0], amloc_cluster.cluster_id,
                                                          gold_amloc_chain))

    strings.append('sent: {}'.format(topic_docs[mention.doc_id].sentences[mention.sent_id].get_raw_sentence()))
    return '\n'.join(strings)


def collect_errors(clusters, other_clusters, topic_docs, is_event):
    '''
    collect event mentions/entity mentions that were clustered incorrectly,
    i.e. where their predicted cluster contained at least 70% of mentions
    that are not in their gold cluster.
    :param clusters: list of Cluster objects of the same type (event/entity clusters)
    :param other_clusters: list of Cluster objects which sets to current entity clusters if clusters are event clusters,
     and vice versa
    :param topic_docs: the current topic's documents
    :param is_event: True if clusters are event clusters and False if they are entity clusters
    :return: set of tuples, when each tuple represents an error.
    '''
    errors = []
    error_ratio = 0.7
    for cluster in clusters:
        mentions_list = []
        for mention in cluster.mentions.values():
            mentions_list.append(mention_data_to_string(mention, other_clusters, is_event, topic_docs))
        cluster_mentions = list(cluster.mentions.values())
        if len(cluster_mentions) > 1:
            for mention_1 in cluster_mentions:
                errors_count = 0
                for mention_2 in cluster_mentions:
                    if mention_1.gold_tag != mention_2.gold_tag:
                        errors_count += 1
                if errors_count/float(len(cluster_mentions)-1) > error_ratio:
                    errors.append((mention_data_to_string(mention_1, other_clusters, is_event, topic_docs)
                                   , mentions_list))

    return errors


def sample_errors(error_list, out_path):
    '''
    Samples 50 errors from error_list
    :param error_list: list of errors collected from each topic
    :param out_path: path to output file
    '''
    random.shuffle(error_list)
    sample = error_list[:50]
    with open(out_path,'w') as f:
        for error in sample:
            f.write('Wrong mention - {}\n'.format(error[0]))
            f.write('cluster: \n')
            for mention in error[1]:
                f.write('{}\n'.format(mention))
                f.write('\n')
            f.write('------------------------------------------------------S')
            f.write('\n')


def mention_to_rep(mention):
    '''
    Returns the mention's representation and its components.
    :param mention: a Mention object (EventMention/EntityMention)
    :return: the mention's representation and its components (tuple of three tensors).
    '''
    span_rep = mention.span_rep
    mention_tensor = torch.squeeze(torch.cat([span_rep, mention.arg0_vec, mention.arg1_vec,
                                  mention.loc_vec, mention.time_vec], 1)).cpu().numpy()
    args_vector = torch.squeeze(torch.cat([mention.arg0_vec, mention.arg1_vec,
                                  mention.loc_vec, mention.time_vec], 1)).cpu().numpy()

    context_vector = mention.head_elmo_embeddings

    return mention_tensor, args_vector , context_vector


def save_mention_representations(clusters, out_dir, is_event):
    '''
    Saves to a pickle file, all mention representations (belong to the test set)
    :param clusters: list of Cluster objects
    :param out_dir: output directory
    :param is_event: True if clusters are event clusters and False if they are entity clusters
    '''
    mention_to_rep_dict = {}
    for cluster in clusters:
        for mention_id, mention in cluster.mentions.items():
            mention_rep = mention_to_rep(mention)
            mention_to_rep_dict[(mention.mention_str, mention.gold_tag)] = mention_rep

    print(len(mention_to_rep_dict))
    filename = 'event_mentions_to_rep_dict' if is_event else 'entity_mentions_to_rep_dict'
    with open(os.path.join(out_dir, filename), 'wb') as f:
        cPickle.dump(mention_to_rep_dict, f)