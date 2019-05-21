import os
import sys
import json
import torch
import argparse
import _pickle as cPickle
import logging

import spacy
from nltk.corpus import wordnet as wn

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

from collections import defaultdict
from swirl_parsing import parse_swirl_output
from allen_srl_reader import read_srl
from create_elmo_embeddings import *
from classes import Document, Sentence, Token, EventMention, EntityMention
from extraction_utils import *


nlp = spacy.load('en')

parser = argparse.ArgumentParser(description='Feature extraction (predicate-argument structures,'
                                             'mention heads, and ELMo embeddings)')

parser.add_argument('--config_path', type=str,
                    help=' The path to the configuration json file')
parser.add_argument('--output_path', type=str,
                    help=' The path to output folder (Where to save the processed data)')

args = parser.parse_args()

out_dir = args.output_path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

with open(os.path.join(args.output_path,'build_features_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)


def load_mentions_from_json(mentions_json_file, docs, is_event, is_gold_mentions):
    '''
    Loading mentions from JSON file and add those to the documents objects
    :param mentions_json_file: the JSON file contains the mentions
    :param docs:  set of document objects
    :param is_event: a boolean indicates whether the function extracts event or entity mentions
    :param is_gold_mentions: a boolean indicates whether the function extracts gold or predicted
    mentions
    '''
    with open(mentions_json_file, 'r') as js_file:
        js_mentions = json.load(js_file)

    for js_mention in js_mentions:
        doc_id = js_mention["doc_id"].replace('.xml', '')
        sent_id = js_mention["sent_id"]
        tokens_numbers = js_mention["tokens_number"]
        mention_type = js_mention["mention_type"]
        is_singleton = js_mention["is_singleton"]
        is_continuous = js_mention["is_continuous"]
        mention_str = js_mention["tokens_str"]
        coref_chain = js_mention["coref_chain"]
        if mention_str is None:
            print(js_mention)
        head_text, head_lemma = find_head(mention_str)
        score = js_mention["score"]
        try:
            token_objects = docs[doc_id].get_sentences()[sent_id].find_mention_tokens(tokens_numbers)
        except:
            print('error when looking for mention tokens')
            print('doc id {} sent id {}'.format(doc_id, sent_id))
            print('token numbers - {}'.format(str(tokens_numbers)))
            print('mention string {}'.format(mention_str))
            print('sentence - {}'.format(docs[doc_id].get_sentences()[sent_id].get_raw_sentence()))
            raise

        # Sanity check - check if all mention's tokens can be found
        if not token_objects:
            print('Can not find tokens of a mention - {} {} {}'.format(doc_id, sent_id,tokens_numbers))

        # Mark the mention's gold coref chain in its tokens
        if is_gold_mentions:
            for token in token_objects:
                if is_event:
                    token.gold_event_coref_chain.append(coref_chain)
                else:
                    token.gold_entity_coref_chain.append(coref_chain)

        if is_event:
            mention = EventMention(doc_id, sent_id, tokens_numbers,token_objects,mention_str, head_text,
                                   head_lemma, is_singleton, is_continuous, coref_chain)
        else:
            mention = EntityMention(doc_id, sent_id, tokens_numbers,token_objects, mention_str, head_text,
                                    head_lemma, is_singleton, is_continuous, coref_chain, mention_type)

        mention.probability = score  # a confidence score for predicted mentions (if used), set gold mentions prob to 1.0
        if is_gold_mentions:
            docs[doc_id].get_sentences()[sent_id].add_gold_mention(mention, is_event)
        else:
            docs[doc_id].get_sentences()[sent_id]. \
                add_predicted_mention(mention, is_event,
                                      relaxed_match=config_dict["relaxed_match_with_gold_mention"])


def load_gold_mentions(docs,events_json, entities_json):
    '''
    A function loads gold event and entity mentions
    :param docs: set of document objects
    :param events_json:  a JSON file contains the gold event mentions (of a specific split - train/dev/test)
    :param entities_json: a JSON file contains the gold entity mentions (of a specific split - train/dev/test)
    '''
    load_mentions_from_json(events_json,docs,is_event=True, is_gold_mentions=True)
    load_mentions_from_json(entities_json,docs,is_event=False, is_gold_mentions=True)


def load_predicted_mentions(docs,events_json, entities_json):
    '''
    This function loads predicted event and entity mentions
    :param docs: set of document objects
    :param events_json:  a JSON file contains predicted event mentions (of a specific split - train/dev/test)
    :param entities_json: a JSON file contains predicted entity mentions (of a specific split - train/dev/test)
    '''
    load_mentions_from_json(events_json,docs,is_event=True, is_gold_mentions=False)
    load_mentions_from_json(entities_json,docs,is_event=False, is_gold_mentions=False)


def load_gold_data(split_txt_file, events_json, entities_json):
    '''
    This function loads the texts of each split and its gold mentions, create document objects
    and stored the gold mentions within their suitable document objects
    :param split_txt_file: the text file of each split is written as 5 columns (stored in data/intermid)
    :param events_json: a JSON file contains the gold event mentions
    :param entities_json: a JSON file contains the gold event mentions
    :return:
    '''
    logger.info('Loading gold mentions...')
    docs = load_ECB_plus(split_txt_file)
    load_gold_mentions(docs, events_json, entities_json)
    return docs


def load_predicted_data(docs, pred_events_json, pred_entities_json):
    '''
    This function loads the predicted mentions and stored them within their suitable document objects
    (suitable for loading the test data)
    :param docs: dictionary that contains the document objects
    :param pred_events_json: a JSON file contains predicted event mentions
    :param pred_entities_json: a JSON file contains predicted event mentions
    :return:
    '''
    logger.info('Loading predicted mentions...')
    load_predicted_mentions(docs, pred_events_json, pred_entities_json)


def find_head(x):
    '''
    This function finds the head and head lemma of a mention x
    :param x: A mention object
    :return: the head word and
    '''

    x_parsed = nlp(x)
    for tok in x_parsed:
        if tok.head == tok:
            if tok.lemma_ == u'-PRON-':
                return tok.text, tok.text.lower()
            return tok.text,tok.lemma_


def have_string_match(mention,arg_str ,arg_start, arg_end):
    '''
    This function checks whether a given entity mention has a string match (strict or relaxed)
    with a span of an extracted argument
    :param mention: a candidate entity mention
    :param arg_str: the argument's text
    :param arg_start: the start index of the argument's span
    :param arg_end: the end index of the argument's span
    :return: True if there is a string match (strict or relaxed) between the entity mention
    and the extracted argument's span, and false otherwise
    '''
    if mention.mention_str == arg_str and mention.start_offset == arg_start:  # exact string match + same start index
        return True
    if mention.mention_str == arg_str:  # exact string match
        return True
    if mention.start_offset >= arg_start and mention.end_offset <= arg_end:  # the argument span contains the mention span
        return True
    if arg_start >= mention.start_offset and arg_end <= mention.end_offset:  # the mention span contains the mention span
        return True
    if len(set(mention.tokens_numbers).intersection(set(range(arg_start,arg_end + 1)))) > 0: # intersection between the mention's tokens and the argument's tokens
        return True
    return False


def add_arg_to_event(entity, event, rel_name):
    '''
    Adds the entity mention as an argument (in a specific role) of an event mention and also adds the
    event mention as predicate (in a specific role) of the entity mention
    :param entity: an entity mention object
    :param event: an event mention object
    :param rel_name: the specific role
    '''
    if rel_name == 'A0':
        event.arg0 = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'A0')
    elif rel_name == 'A1':
        event.arg1 = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'A1')
    elif rel_name == 'AM-TMP':
        event.amtmp = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'AM-TMP')
    elif rel_name == 'AM-LOC':
        event.amloc = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'AM-LOC')


def find_argument(rel_name, rel_tokens, matched_event, sent_entities, sent_obj, is_gold, srl_obj):
    '''
    This function matches between an argument of an event mention and an entity mention.
    :param rel_name: the specific role of the argument
    :param rel_tokens: the argument's tokens
    :param matched_event: the event mention
    :param sent_entities: a entity mentions exist in the event's sentence.
    :param sent_obj: the object represents the sentence
    :param is_gold: whether the argument need to be matched with a gold mention or not
    :param srl_obj: an object represents the extracted SRL argument.
    :return True if the extracted SRL argument was matched with an entity mention.
    '''
    arg_start_ix = rel_tokens[0]
    if len(rel_tokens) > 1:
        arg_end_ix = rel_tokens[1]
    else:
        arg_end_ix = rel_tokens[0]

    if arg_end_ix >= len(sent_obj.get_tokens()):
        print('argument bound mismatch with sentence length')
        print('arg start index - {}'.format(arg_start_ix))
        print('arg end index - {}'.format(arg_end_ix))
        print('sentence length - {}'.format(len(sent_obj.get_tokens())))
        print('raw sentence: {}'.format(sent_obj.get_raw_sentence()))
        print('matched event: {}'.format(str(matched_event)))
        print('srl obj - {}'.format(str(srl_obj)))

    arg_str, arg_tokens = sent_obj.fetch_mention_string(arg_start_ix, arg_end_ix)

    entity_found = False
    matched_entity = None
    for entity in sent_entities:
        if have_string_match(entity, arg_str, arg_start_ix, arg_end_ix):
            if rel_name == 'AM-TMP' and entity.mention_type != 'TIM':
                continue
            if rel_name == 'AM-LOC' and entity.mention_type != 'LOC':
                continue
            entity_found = True
            matched_entity = entity
            break
    if entity_found:
        add_arg_to_event(matched_entity, matched_event, rel_name)
        if is_gold:
            return True
        else:
            if matched_entity.gold_mention_id is not None:
                return True
            else:
                return False
    else:
        return False


def match_allen_srl_structures(dataset, srl_data, is_gold):
    '''
    Matches between extracted predicates and event mentions and between their arguments and
    entity mentions, designed to handle the output of Allen NLP SRL system
    :param dataset: an object represents the spilt (train/dev/test)
    :param srl_data: a dictionary contains the predicate-argument structures
    :param is_gold: whether to match predicate-argument structures with gold mentions or with predicted mentions
    '''
    matched_events_count = 0
    matched_args_count = 0

    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                # Handling nominalizations in case we don't use syntactic dependencies (which already handle this)
                if not config_dict["use_dep"]:
                    sent_str = sent.get_raw_sentence()
                    parsed_sent = nlp(sent_str)
                    find_nominalizations_args(parsed_sent, sent, is_gold)
                sent_srl_info = None

                if doc_id in srl_data:
                    doc_srl = srl_data[doc_id]
                    if int(sent_id) in doc_srl:
                        sent_srl_info = doc_srl[int(sent_id)]

                if sent_srl_info is not None:
                    for event_srl in sent_srl_info.srl:
                        event_text = event_srl.verb.text
                        event_ecb_tok_ids = event_srl.verb.ecb_tok_ids

                        if is_gold:
                            sent_events = sent.gold_event_mentions
                            sent_entities = sent.gold_entity_mentions
                        else:
                            sent_events = sent.pred_event_mentions
                            sent_entities = sent.pred_entity_mentions
                        event_found = False
                        matched_event = None

                        for event_mention in sent_events:
                            if event_ecb_tok_ids == event_mention.tokens_numbers or \
                                    event_text == event_mention.mention_str or \
                                    event_text in event_mention.mention_str or \
                                    event_mention.mention_str in event_text:
                                event_found = True
                                matched_event = event_mention
                                if is_gold:
                                    matched_events_count += 1
                                elif matched_event.gold_mention_id is not None:
                                    matched_events_count += 1
                            if event_found:
                                break
                        if event_found:
                            if event_srl.arg0 is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg0, 'A0', is_gold):
                                    matched_args_count += 1

                            if event_srl.arg1 is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg1, 'A1', is_gold):
                                    matched_args_count += 1
                            if event_srl.arg_tmp is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg_tmp, 'AM-TMP', is_gold):
                                    matched_args_count += 1

                            if event_srl.arg_loc is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg_loc, 'AM-LOC', is_gold):
                                    matched_args_count += 1

    logger.info('SRL matched events - ' + str(matched_events_count))
    logger.info('SRL matched args - ' + str(matched_args_count))


def match_entity_with_srl_argument(sent_entities, matched_event ,srl_arg,rel_name, is_gold):
    '''
    This function matches between an argument of an event mention and an entity mention.
    Designed to handle the output of Allen NLP SRL system
    :param sent_entities: the entity mentions in the event's sentence
    :param matched_event: the event mention
    :param srl_arg: the extracted argument
    :param rel_name: the role name
    :param is_gold: whether to match the argument with gold entity mention or with predicted entity mention
    :return:
    '''
    found_entity = False
    matched_entity = None
    for entity in sent_entities:
        if srl_arg.ecb_tok_ids == entity.tokens_numbers or \
                srl_arg.text == entity.mention_str or \
                srl_arg.text in entity.mention_str or \
                entity.mention_str in srl_arg.text:
            if rel_name == 'AM-TMP' and entity.mention_type != 'TIM':
                continue
            if rel_name == 'AM-LOC' and entity.mention_type != 'LOC':
                continue
            found_entity = True
            matched_entity = entity

        if found_entity:
            break

    if found_entity:
        add_arg_to_event(matched_entity, matched_event, rel_name)
        if is_gold:
            return True
        else:
            if matched_entity.gold_mention_id is not None:
                return True
            else:
                return False
    else:
        return False


def load_srl_info(dataset, srl_data, is_gold):
    '''
    Matches between extracted predicates and event mentions and between their arguments and
    entity mentions.
    :param dataset: an object represents the spilt (train/dev/test)
    :param srl_data: a dictionary contains the predicate-argument structures
    :param is_gold: whether to match predicate-argument structures with gold mentions or with predicted mentions
    '''
    matched_events_count = 0
    unmatched_event_count = 0
    matched_args_count = 0

    matched_identified_events = 0
    matched_identified_args = 0
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                # Handling nominalizations if we don't use dependency parsing (that already handles it)
                if not config_dict["use_dep"]:
                    sent_str = sent.get_raw_sentence()
                    parsed_sent = nlp(sent_str)
                    find_nominalizations_args(parsed_sent, sent, is_gold)
                sent_srl_info = {}

                if doc_id in srl_data:
                    doc_srl = srl_data[doc_id]
                    if int(sent_id) in doc_srl:
                        sent_srl_info = doc_srl[int(sent_id)]
                else:
                    print('doc not in srl data - ' + doc_id)

                for event_key, srl_obj in sent_srl_info.items():
                    if is_gold:
                        sent_events = sent.gold_event_mentions
                        sent_entities = sent.gold_entity_mentions
                    else:
                        sent_events = sent.pred_event_mentions
                        sent_entities = sent.pred_entity_mentions
                    event_found = False
                    matched_event = None
                    for event_mention in sent_events:
                        if event_key in event_mention.tokens_numbers:
                            event_found = True
                            matched_event = event_mention
                            if is_gold:
                                matched_events_count += 1
                            elif matched_event.gold_mention_id is not None:
                                    matched_events_count += 1
                        if event_found:
                            break
                    if event_found:
                        for rel_name, rel_tokens in srl_obj.arg_info.items():
                            if find_argument(rel_name, rel_tokens, matched_event, sent_entities, sent, is_gold,srl_obj):
                                matched_args_count += 1
                    else:
                        unmatched_event_count += 1
    logger.info('SRL matched events - ' + str(matched_events_count))
    logger.info('SRL unmatched events - ' + str(unmatched_event_count))
    logger.info('SRL matched args - ' + str(matched_args_count))


def find_topic_gold_clusters(topic):
    '''
    Finds the gold clusters of a specific topic
    :param topic: a topic object
    :return: a mapping of coref chain to gold cluster (for a specific topic) and the topic's mentions
    '''
    event_mentions = []
    entity_mentions = []
    # event_gold_tag_to_cluster = defaultdict(list)
    # entity_gold_tag_to_cluster = defaultdict(list)

    event_gold_tag_to_cluster = {}
    entity_gold_tag_to_cluster = {}

    for doc_id, doc in topic.docs.items():
        for sent_id, sent in doc.sentences.items():
            event_mentions.extend(sent.gold_event_mentions)
            entity_mentions.extend(sent.gold_entity_mentions)

    for event in event_mentions:
        if event.gold_tag != '-':
            if event.gold_tag not in event_gold_tag_to_cluster:
                event_gold_tag_to_cluster[event.gold_tag] = []
            event_gold_tag_to_cluster[event.gold_tag].append(event)
    for entity in entity_mentions:
        if entity.gold_tag != '-':
            if entity.gold_tag not in entity_gold_tag_to_cluster:
                entity_gold_tag_to_cluster[entity.gold_tag] = []
            entity_gold_tag_to_cluster[entity.gold_tag].append(entity)

    return event_gold_tag_to_cluster, entity_gold_tag_to_cluster, event_mentions, entity_mentions


def write_dataset_statistics(split_name, dataset, check_predicted):
    '''
    Prints the split statistics
    :param split_name: the split name (a string)
    :param dataset: an object represents the split
    :param check_predicted: whether to print statistics of predicted mentions too
    '''
    docs_count = 0
    sent_count = 0
    event_mentions_count = 0
    entity_mentions_count = 0
    event_chains_count = 0
    entity_chains_count = 0
    topics_count = len(dataset.topics.keys())
    predicted_events_count = 0
    predicted_entities_count = 0
    matched_predicted_event_count = 0
    matched_predicted_entity_count = 0


    for topic_id, topic in dataset.topics.items():
        event_gold_tag_to_cluster, entity_gold_tag_to_cluster, \
        event_mentions, entity_mentions = find_topic_gold_clusters(topic)

        docs_count += len(topic.docs.keys())
        sent_count += sum([len(doc.sentences.keys()) for doc_id, doc in topic.docs.items()])
        event_mentions_count += len(event_mentions)
        entity_mentions_count += len(entity_mentions)

        entity_chains = set()
        event_chains = set()

        for mention in entity_mentions:
            entity_chains.add(mention.gold_tag)

        for mention in event_mentions:
            event_chains.add(mention.gold_tag)

        # event_chains_count += len(set(event_gold_tag_to_cluster.keys()))
        # entity_chains_count += len(set(entity_gold_tag_to_cluster.keys()))

        event_chains_count += len(event_chains)
        entity_chains_count += len(entity_chains)

        if check_predicted:
            for doc_id, doc in topic.docs.items():
                for sent_id, sent in doc.sentences.items():
                    pred_events = sent.pred_event_mentions
                    pred_entities = sent.pred_entity_mentions

                    predicted_events_count += len(pred_events)
                    predicted_entities_count += len(pred_entities)

                    for pred_event in pred_events:
                        if pred_event.has_compatible_mention:
                            matched_predicted_event_count += 1

                    for pred_entity in pred_entities:
                        if pred_entity.has_compatible_mention:
                            matched_predicted_entity_count += 1

    with open(os.path.join(args.output_path, '{}_statistics.txt'.format(split_name)), 'w') as f:
        f.write('Number of topics - {}\n'.format(topics_count))
        f.write('Number of documents - {}\n'.format(docs_count))
        f.write('Number of sentences - {}\n'.format(sent_count))
        f.write('Number of event mentions - {}\n'.format(event_mentions_count))
        f.write('Number of entity mentions - {}\n'.format(entity_mentions_count))

        if check_predicted:
            f.write('Number of predicted event mentions  - {}\n'.format(predicted_events_count))
            f.write('Number of predicted entity mentions - {}\n'.format(predicted_entities_count))
            f.write('Number of predicted event mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_event_count,
                                        (matched_predicted_event_count/float(event_mentions_count)) *100 ))
            f.write('Number of predicted entity mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_entity_count,
                                        (matched_predicted_entity_count / float(entity_mentions_count)) * 100))


def obj_dict(obj):
    obj_d = obj.__dict__
    obj_d = stringify_keys(obj_d)
    return obj_d


def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    pass

            # delete old key
            del d[key]
    return d


def set_elmo_embed_to_mention(mention, sent_embeddings):
    '''
    Sets the ELMo embeddings of a mention
    :param mention: event/entity mention object
    :param sent_embeddings: the embedding for each word in the sentence produced by ELMo model
    :return:
    '''
    head_index = mention.get_head_index()
    head_embeddings = sent_embeddings[int(head_index)]
    mention.head_elmo_embeddings = torch.from_numpy(head_embeddings)


def set_elmo_embeddings_to_mentions(elmo_embedder, sentence, set_pred_mentions):
    '''
     Sets the ELMo embeddings for all the mentions in the sentence
    :param elmo_embedder: a wrapper object for ELMo model of Allen NLP
    :param sentence: a sentence object
    '''
    avg_sent_embeddings = elmo_embedder.get_elmo_avg(sentence)
    event_mentions = sentence.gold_event_mentions
    entity_mentions = sentence.gold_entity_mentions

    for event in event_mentions:
        set_elmo_embed_to_mention(event,avg_sent_embeddings)

    for entity in entity_mentions:
        set_elmo_embed_to_mention(entity, avg_sent_embeddings)

    # Set the contextualized vector also for predicted mentions
    if set_pred_mentions:
        event_mentions = sentence.pred_event_mentions
        entity_mentions = sentence.pred_entity_mentions

        for event in event_mentions:
            set_elmo_embed_to_mention(event, avg_sent_embeddings)  # set the head contextualized vector

        for entity in entity_mentions:
            set_elmo_embed_to_mention(entity, avg_sent_embeddings)  # set the head contextualized vector


def load_elmo_embeddings(dataset, elmo_embedder, set_pred_mentions):
    '''
    Sets the ELMo embeddings for all the mentions in the split
    :param dataset: an object represents a split (train/dev/test)
    :param elmo_embedder: a wrapper object for ELMo model of Allen NLP
    :return:
    '''
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                set_elmo_embeddings_to_mentions(elmo_embedder, sent, set_pred_mentions)


def main(args):
    """
        This script loads the train, dev and test json files (contain the gold entity and event
        mentions) builds mention objects, extracts predicate-argument structures, mention head
        and ELMo embeddings for each mention.

        Runs data processing scripts to turn intermediate data from (../intermid) into
        processed data ready to use in training and inference(saved in ../processed).
    """
    logger.info('Training data - loading event and entity mentions')
    training_data = load_gold_data(config_dict["train_text_file"],config_dict["train_event_mentions"],
                                   config_dict["train_entity_mentions"])

    logger.info('Dev data - Loading event and entity mentions ')
    dev_data = load_gold_data(config_dict["dev_text_file"],config_dict["dev_event_mentions"],
                              config_dict["dev_entity_mentions"])

    logger.info('Test data - Loading event and entity mentions')
    test_data = load_gold_data(config_dict["test_text_file"], config_dict["test_event_mentions"],
                               config_dict["test_entity_mentions"])

    if config_dict["load_predicted_mentions"]:
        load_predicted_data(test_data, config_dict["pred_event_mentions"], config_dict["pred_entity_mentions"])

    train_set = order_docs_by_topics(training_data)
    dev_set = order_docs_by_topics(dev_data)
    test_set = order_docs_by_topics(test_data)

    write_dataset_statistics('train', train_set, check_predicted=False)

    write_dataset_statistics('dev', dev_set, check_predicted=False)

    check_predicted = True if config_dict["load_predicted_mentions"] else False
    write_dataset_statistics('test', test_set, check_predicted=check_predicted)

    if config_dict["use_srl"]:
        logger.info('Loading SRL info')
        if config_dict["use_allen_srl"]: # use the SRL system which is implemented in AllenNLP (currently - a deep BiLSTM model (He et al, 2017).)
            srl_data = read_srl(config_dict["srl_output_path"])
            logger.info('Training gold mentions - loading SRL info')
            match_allen_srl_structures(train_set, srl_data, is_gold=True)
            logger.info('Dev gold mentions - loading SRL info')
            match_allen_srl_structures(dev_set, srl_data, is_gold=True)
            logger.info('Test gold mentions - loading SRL info')
            match_allen_srl_structures(test_set, srl_data, is_gold=True)
            if config_dict["load_predicted_mentions"]:
                logger.info('Test predicted mentions - loading SRL info')
                match_allen_srl_structures(test_set, srl_data, is_gold=False)
        else: # Use SwiRL SRL system (Surdeanu et al., 2007)
            srl_data = parse_swirl_output(config_dict["srl_output_path"])
            logger.info('Training gold mentions - loading SRL info')
            load_srl_info(train_set, srl_data, is_gold=True)
            logger.info('Dev gold mentions - loading SRL info')
            load_srl_info(dev_set, srl_data, is_gold=True)
            logger.info('Test gold mentions - loading SRL info')
            load_srl_info(test_set, srl_data, is_gold=True)
            if config_dict["load_predicted_mentions"]:
                logger.info('Test predicted mentions - loading SRL info')
                load_srl_info(test_set, srl_data, is_gold=False)
    if config_dict["use_dep"]:  # use dependency parsing
        logger.info('Augmenting predicate-arguments structures using dependency parser')
        logger.info('Training gold mentions - loading predicates and their arguments with dependency parser')
        find_args_by_dependency_parsing(train_set, is_gold=True)
        logger.info('Dev gold mentions - loading predicates and their arguments with dependency parser')
        find_args_by_dependency_parsing(dev_set, is_gold=True)
        logger.info('Test gold mentions - loading predicates and their arguments with dependency parser')
        find_args_by_dependency_parsing(test_set, is_gold=True)
        if config_dict["load_predicted_mentions"]:
            logger.info('Test predicted mentions - loading predicates and their arguments with dependency parser')
            find_args_by_dependency_parsing(test_set, is_gold=False)
    if config_dict["use_left_right_mentions"]:  # use left and right mentions heuristic
        logger.info('Augmenting predicate-arguments structures using leftmost and rightmost entity mentions')
        logger.info('Training gold mentions - loading predicates and their arguments ')
        find_left_and_right_mentions(train_set, is_gold=True)
        logger.info('Dev gold mentions - loading predicates and their arguments ')
        find_left_and_right_mentions(dev_set, is_gold=True)
        logger.info('Test gold mentions - loading predicates and their arguments ')
        find_left_and_right_mentions(test_set, is_gold=True)
        if config_dict["load_predicted_mentions"]:
            logger.info('Test predicted mentions - loading predicates and their arguments ')
            find_left_and_right_mentions(test_set, is_gold=False)

    if config_dict["load_elmo"]: # load ELMo embeddings
        elmo_embedder = ElmoEmbedding(config_dict["options_file"], config_dict["weight_file"])
        logger.info("Loading ELMO embeddings...")
        load_elmo_embeddings(train_set, elmo_embedder, set_pred_mentions=False)
        load_elmo_embeddings(dev_set, elmo_embedder, set_pred_mentions=False)
        load_elmo_embeddings(test_set, elmo_embedder, set_pred_mentions=True)

    logger.info('Storing processed data...')
    with open(os.path.join(args.output_path,'training_data'), 'wb') as f:
        cPickle.dump(train_set, f)
    with open(os.path.join(args.output_path,'dev_data'), 'wb') as f:
        cPickle.dump(dev_set, f)
    with open(os.path.join(args.output_path, 'test_data'), 'wb') as f:
        cPickle.dump(test_set, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main(args)
