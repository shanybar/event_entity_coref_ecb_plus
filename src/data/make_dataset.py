# -*- coding: utf-8 -*-
import os
import csv
import json
import _pickle as cPickle
import logging
import argparse
from mention_data import MentionData
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Parsing ECB+ corpus')

parser.add_argument('--ecb_path', type=str,
                    help=' The path to the ECB+ corpus')
parser.add_argument('--output_dir', type=str,
                        help=' The directory of the output files')
parser.add_argument('--data_setup', type=int,
                        help='Set the desirable dataset setup, 1 for Yang/Choubey setup and 2 for Cybulska/Kenyon-Dean setup (recommended)')
parser.add_argument('--selected_sentences_file', type=str,
                    help=' The path to a file contains selected sentences from the ECB+ corpus (relevant only for '
                         'the second evaluation setup (Cybulska setup)')
args = parser.parse_args()

out_dir = args.output_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

class Token(object):
    '''
    An helper class which represents a single when reading the corpus.
    '''
    def __init__(self, text, sent_id, tok_id, rel_id=None):
        '''

        :param text: The token text
        :param sent_id: The sentence id
        :param tok_id: The token id
        :param rel_id: The relation id (i.e. coreference chain)
        '''

        self.text = text
        self.sent_id = sent_id
        self.tok_id = tok_id
        self.rel_id = rel_id


def read_selected_sentences(filename):
    '''
    This function reads the CSV file that was released with ECB+ corpus and returns a
    dictionary contains those sentences IDs. This file contains the IDs of 1840 sentences
    which were manually reviewed and checked for correctness.
    The ECB+ creators recommend to use this subset of the dataset.
    :param filename: the CSV file
    :return: a dictionary, where a key is an XML filename (i.e. ECB+ document) and the value is a
    list contains all the sentences IDs that were selected from that XML filename.
    '''
    xml_to_sent_dict = {}
    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        reader.next()
        for line in reader:
            xml_filename = '{}_{}.xml'.format(line[0],line[1])
            sent_id = int(line[2])

            if xml_filename not in xml_to_sent_dict:
                xml_to_sent_dict[xml_filename] = []
            xml_to_sent_dict[xml_filename].append(sent_id)

    return xml_to_sent_dict


def find_mention_class(tag):
    '''
    Given a string represents a mention type, this function returns its abbreviation
    :param tag:  a string represents a mention type
    :return: Abbreviation of the mention type (a string)
    '''
    if 'ACTION' in tag:
        return 'ACT'
    if 'LOC' in tag:
        return 'LOC'
    if 'NON' in tag:
        return 'NON'
    if 'HUMAN' in tag:
        return 'HUM'
    if 'TIME' in tag:
        return 'TIM'
    else:
        print(tag)


def coref_chain_id_to_mention_type(coref_chain_id):
    '''
    Given a string represents a mention's coreference chain ID,
    this function returns a string that represents the mention type.
    :param coref_chain_id: a string represents a mention's coreference chain ID
    :return: a string that represents the mention type
    '''
    if 'ACT' in coref_chain_id or 'NEG' in coref_chain_id:
        return 'ACT'
    if 'LOC' in coref_chain_id:
        return 'LOC'
    if 'NON' in coref_chain_id:
        return 'NON'
    if 'HUM' in coref_chain_id or 'CON' in coref_chain_id:
        return 'HUM'
    if 'TIM' in coref_chain_id:
        return 'TIM'
    if 'UNK' in coref_chain_id:
        return 'UNK'


def calc_split_statistics(dataset_split, split_name, statistics_file_name):
    '''
    This function calculates and saves the statistics of a split (train/dev/test) into a file.
    :param dataset_split: a list that contains all the mention objects in the split
    :param split_name: the split name (a string)
    :param statistics_file_name: a filename for the statistics file
    '''
    event_mentions_count = 0
    human_mentions_count = 0
    non_human_mentions_count = 0
    loc_mentions_count = 0
    time_mentions_count = 0
    non_continuous_mentions_count = 0
    unk_coref_mentions_count = 0
    coref_chains_dict = {}

    for mention_obj in dataset_split:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type or 'NEG' in mention_type:
            event_mentions_count += 1
        elif 'NON' in mention_type:
            non_human_mentions_count += 1
        elif 'HUM' in mention_type:
            human_mentions_count += 1
        elif 'LOC' in mention_type:
            loc_mentions_count += 1
        elif 'TIM' in mention_type:
            time_mentions_count += 1
        else:
            print(mention_type)

        is_continuous = mention_obj.is_continuous
        if not is_continuous:
            non_continuous_mentions_count += 1

        coref_chain = mention_obj.coref_chain
        if 'UNK' in coref_chain:
            unk_coref_mentions_count += 1
        if coref_chain not in coref_chains_dict:
            coref_chains_dict[coref_chain] = 1
    with open(statistics_file_name, 'a') as f:

        f.write('{} statistics\n'.format(split_name))
        f.write('-------------------------\n')
        f.write( 'Number of event mentions - {}\n'.format(event_mentions_count))
        f.write( 'Number of human participants mentions - {}\n'.format(human_mentions_count))
        f.write( 'Number of non-human participants mentions - {}\n'.format(non_human_mentions_count))
        f.write( 'Number of location mentions - {}\n'.format(loc_mentions_count))
        f.write( 'Number of time mentions - {}\n'.format(time_mentions_count))
        f.write( 'Total number of mentions - {}\n'.format(len(dataset_split)))

        f.write( 'Number of non-continuous mentions - {}\n'.format(non_continuous_mentions_count))
        f.write( 'Number of mentions with coref id = UNK - {}\n'.format(unk_coref_mentions_count))
        f.write( 'Number of coref chains = {}\n'.format(len(coref_chains_dict)))
        f.write('\n')


def save_gold_mention_statistics(train_extracted_mentions, dev_extracted_mentions,
                                  test_extracted_mentions):
    '''
    This function calculates and saves the statistics of each split (train/dev/test) into a file.
    :param train_extracted_mentions: a list that contains all the mention objects in the train split
    :param dev_extracted_mentions: a list that contains all the mention objects in the dev split
    :param test_extracted_mentions: a list that contains all the mention objects in the test split
    '''
    logger.info('Calculate mention statistics...')

    all_data_mentions = train_extracted_mentions + dev_extracted_mentions +test_extracted_mentions
    filename = 'mention_stats.txt'
    calc_split_statistics(train_extracted_mentions, 'Train set',
                          os.path.join(args.output_dir,filename))

    calc_split_statistics(dev_extracted_mentions, 'Dev set',
                            os.path.join(args.output_dir, filename))

    calc_split_statistics(test_extracted_mentions, 'Test set',
                          os.path.join(args.output_dir, filename))

    calc_split_statistics(all_data_mentions, 'Total',
                          os.path.join(args.output_dir, filename))

    logger.info('Save mention statistics...')


def read_ecb_plus_doc(selected_sent_list, doc_filename, doc_id, file_obj,extracted_mentions,
                       parse_all, load_singletons):
    '''
    This function reads an ECB+ XML file (i.e. document), extracts its gold mentions and texts.
    the text file of each split is written as 5 columns -  the first column contains the document
    ID (which has the following format - {topic id}_{docuemnt id}_{ecb/ecbplus type}). Note that the topic id
    and the document type determines the sub-topic id, e.g. a suc-topic with topic_id = 1 and
    document_type = ecbplus is different from a sub-topic with topic_id = 1 and document_type = ecb.
    the second column cotains the sentence id, the third column contains the token id,
    and the fourth column contains the token string. The fifth column should be ignored.
    :param selected_sent_list: the selected sentences to extract
    :param doc_filename: the ECB+ file to extract
    :param doc_id: the document ID
    :param file_obj: a file object for writing the document text.
    :param extracted_mentions: a list of split's extracted mentions
    :param parse_all: a boolean variable indicates whether to read all the ECB+ corpus as in
    Yang setup or whether to filter the sentences according to a selected sentences list
    as in Cybulska setup.
    :param load_singletons:  a boolean variable indicates whether to read singleton mentions as in
    Cybulska setup or whether to ignore them as in Yang setup.
    '''
    ecb_file = open(doc_filename, 'r')
    tree = ET.parse(ecb_file)
    root = tree.getroot()

    related_events = {}
    within_coref = {}
    mid_to_tid_dict = {}
    mid_to_event_tag_dict = {}
    mid_to_tag = {}
    tokens = {}
    cur_mid = ''

    mid_to_coref_chain = {}

    for action in root.find('Markables').iter():
        if action.tag == 'Markables':
            continue
        elif action.tag == 'token_anchor':
            mid_to_tid_dict[cur_mid].append(action.attrib['t_id'])
        else:
            cur_mid = action.attrib['m_id']

            if 'TAG_DESCRIPTOR' in action.attrib:
                if 'instance_id' in action.attrib:
                    mid_to_event_tag_dict[cur_mid] = (
                        action.attrib['TAG_DESCRIPTOR'], action.attrib['instance_id'])
                else:
                    mid_to_event_tag_dict[cur_mid] = (action.attrib['TAG_DESCRIPTOR'], action.tag) #intra doc coref
            else:
                mid_to_tid_dict[cur_mid] = []
                mid_to_tag[cur_mid] = action.tag

    cur_instance_id = ''
    source_ids = []
    mapped_mid = []

    for within_doc_coref in root.find('Relations').findall('INTRA_DOC_COREF'):
        for child in within_doc_coref.iter():
            if child.tag == 'INTRA_DOC_COREF':

                mention_coref_class = mid_to_event_tag_dict[within_doc_coref.find('target').get('m_id')][1] # find the mention class of intra doc coref mention
                if mention_coref_class == 'UNKNOWN_INSTANCE_TAG':
                    cls = 'UNK'
                else:
                    mention_class = mid_to_tag[within_doc_coref.find('source').get('m_id')]
                    cls = find_mention_class(mention_class)
                cur_instance_id = 'INTRA_{}_{}_{}'.format(cls,child.attrib['r_id'],doc_id)
                within_coref[cur_instance_id] = ()
            else:
                if child.tag == 'source':
                    source_ids += (mid_to_tid_dict[child.attrib['m_id']])
                    mapped_mid.append(child.attrib['m_id'])
                    mid_to_coref_chain[child.attrib['m_id']] = cur_instance_id
                else:
                    within_coref[cur_instance_id] = (source_ids, mid_to_event_tag_dict[child.attrib['m_id']][0])
                    source_ids = []

    cur_instance_id = ''
    source_ids = []

    for cross_doc_coref in root.find('Relations').findall('CROSS_DOC_COREF'):
        for child in cross_doc_coref.iter():
            if child.tag == 'CROSS_DOC_COREF':
                related_events[child.attrib['note']] = ()
                cur_instance_id = child.attrib['note']
            else:
                if child.tag == 'source':
                    source_ids += (mid_to_tid_dict[child.attrib['m_id']])
                    mapped_mid.append(child.attrib['m_id'])
                    mid_to_coref_chain[child.attrib['m_id']] = cur_instance_id
                else:
                    related_events[cur_instance_id] = (
                        source_ids, mid_to_event_tag_dict[child.attrib['m_id']][0])
                    source_ids = []

    for token in root.findall('token'):
        tokens[token.attrib['t_id']] = Token(token.text, token.attrib['sentence'], token.attrib['number'])

    for key in related_events:
        for token_id in related_events[key][0]:
            tokens[token_id].rel_id = (key, related_events[key][1])
    for key in within_coref:
        for token_id in within_coref[key][0]:
            tokens[token_id].rel_id = (key, within_coref[key][1])

    if load_singletons:
        # Singletons - find the mention m_ids that are not mapped to any coref chain and create instance id for each mention
        for mid in mid_to_tid_dict:
            if mid not in mapped_mid:  # singleton mention
                mention_class = mid_to_tag[mid]
                cls = find_mention_class(mention_class)
                singleton_instance_id = 'Singleton_{}_{}_{}'.format(cls,mid,doc_id )
                mid_to_coref_chain[mid] = singleton_instance_id
                unmapped_tids = mid_to_tid_dict[mid]
                for token_id in unmapped_tids:
                    if tokens[token_id].rel_id is None:
                        tokens[token_id].rel_id = (singleton_instance_id,'padding')

    # creating an instance for each mention
    for mid in mid_to_tid_dict:
        tids = mid_to_tid_dict[mid]
        token_numbers = []  # the ordinal token numbers of each token in its sentence
        tokens_str = []
        sent_id = None

        if mid not in mid_to_coref_chain:
            continue
        coref_chain = mid_to_coref_chain[mid]
        type_tag = mid_to_tag[mid]
        mention_type = find_mention_class(type_tag)

        mention_type_by_coref_chain = coref_chain_id_to_mention_type(coref_chain)
        if mention_type != mention_type_by_coref_chain:
            print('coref chain: {}'.format(coref_chain))
            print('mention type by coref chain: {}'.format(mention_type_by_coref_chain))
            print('mention type: {}'.format(mention_type))

        for token_id in tids:
            token = tokens[token_id]
            if sent_id is None:
                sent_id = int(token.sent_id)

            if int(token.tok_id) not in token_numbers:
                token_numbers.append(int(token.tok_id))
                tokens_str.append(token.text.encode('ascii', 'ignore'))

        is_continuous = True if token_numbers == range(token_numbers[0], token_numbers[-1]+1) else False
        is_singleton = True if 'Singleton' in coref_chain else False
        if parse_all or sent_id in selected_sent_list:
            if 'plus' in doc_id:
                if sent_id > 0:
                    sent_id -= 1
            if parse_all and (doc_id == '9_3ecbplus' or doc_id == '9_4ecbplus'):
                if sent_id > 0:
                    sent_id -= 1

            mention_obj = MentionData(doc_id, sent_id, token_numbers, ' '.join(tokens_str),
                                       coref_chain, mention_type,is_continuous=is_continuous,
                                       is_singleton=is_singleton, score=float(-1))
            extracted_mentions.append(mention_obj)
    prev_sent_id = None

    for token in root.findall('token'):
        token = tokens[token.attrib['t_id']]
        token_id = int(token.tok_id)
        sent_id = int(token.sent_id)
        if not parse_all and sent_id not in selected_sent_list:
            continue
        if 'plus' in doc_id:
            if sent_id > 0:
                sent_id -= 1
                if parse_all and (doc_id == '9_3ecbplus' or doc_id == '9_4ecbplus'):
                    if sent_id == 0:
                        continue
                    else:
                        sent_id -= 1
            else:
                continue

        if prev_sent_id is None or prev_sent_id != sent_id:
            file_obj.write('\n')
            prev_sent_id = sent_id
        text = token.text.encode('ascii', 'ignore')

        if text == '' or text == '\t':
            text = '-'

        if token.rel_id is not None:
            file_obj.write(doc_id + '\t' + str(sent_id) + '\t' + str(token_id) + '\t' + text + '\t' + \
                            token.rel_id[0] + '\n')
        else:
            file_obj.write(doc_id + '\t' + str(sent_id) + '\t' + str(token_id) + '\t' + text + '\t-' + '\n')


def obj_dict(obj):
    return obj.__dict__


def save_split_mentions_to_json(split_name, mentions_list):
    '''
    This function gets a mentions list of a specific split and saves its mentions in a JSON files.
    Note that event and entity mentions are saved in separate files.
    :param split_name: the split name
    :param mentions_list: the split's extracted mentions list
    '''
    event_mentions = []
    entity_mentions = []

    for mention_obj in mentions_list:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type or 'NEG' in mention_type:
            event_mentions.append(mention_obj)
        else:
            entity_mentions.append(mention_obj)

    json_event_filename = os.path.join(args.output_dir, 'ECB_{}_Event_gold_mentions.json'.format(split_name))
    json_entity_filename =  os.path.join(args.output_dir, 'ECB_{}_Entity_gold_mentions.json'.format(split_name))

    with open(json_event_filename, 'w') as f:
        json.dump(event_mentions, f, default=obj_dict, indent=4, sort_keys=True)

    with open(json_entity_filename, 'w') as f:
        json.dump(entity_mentions, f, default=obj_dict, indent=4, sort_keys=True)


def parse_selected_sentences(xml_to_sent_dict, parse_all, load_singletons,data_setup):
    '''

    :param xml_to_sent_dict: selected sentences dictionary
    :param parse_all: a boolean variable indicates whether to read all the ECB+ corpus as in
    Yang setup or whether to filter the sentences according to a selected sentences list
    as in Cybulska setup.
    :param load_singletons:  boolean variable indicates whether to read singleton mentions as in
    Cybulska setup or whether to ignore them as in Yang setup.
    :param data_setup: the variable indicates the evaluation setup -
     1 for Yang and Choubey setup and 2 for Cybulska Kenyon-Dean setup (recommended).
    '''
    if data_setup == 1:  # Yang setup
        train_topics = range(1,23)
        dev_topics = range(23,26)
    else:  # Cybulska setup
        dev_topics = [2, 5, 12, 18, 21, 23, 34, 35]
        train_topics = [i for i in range(1,36) if i not in dev_topics]  # train topics 1-35 , test topics 36-45

    dev_out = open(os.path.join(args.output_dir, 'ECB_Dev_corpus.txt'), 'w')
    train_out = open(os.path.join(args.output_dir, 'ECB_Train_corpus.txt'), 'w')
    test_out = open(os.path.join(args.output_dir, 'ECB_Test_corpus.txt'), 'w')

    dirs = os.listdir(args.ecb_path)
    dirs_int = [int(dir) for dir in dirs]
    train_ecb_files_sorted = []
    test_ecb_files_sorted = []
    train_ecb_plus_files_sorted = []
    test_ecb_plus_files_sorted = []
    dev_ecb_files_sorted = []
    dev_ecb_plus_files_sorted = []

    for topic in sorted(dirs_int):
        dir = str(topic)

        doc_files = os.listdir(os.path.join(args.ecb_path,dir))
        ecb_files = []
        ecb_plus_files = []
        for doc_file in doc_files:

            if 'plus' in doc_file:
                ecb_plus_files.append(doc_file)
            else:
                ecb_files.append(doc_file)

        ecb_files = sorted(ecb_files)
        ecb_plus_files=sorted(ecb_plus_files)

        for ecb_file in ecb_files:
            if parse_all or ecb_file in xml_to_sent_dict:
                xml_filename = os.path.join(os.path.join(args.ecb_path,dir),ecb_file)
                if parse_all:
                    selected_sentences = None
                else:
                    selected_sentences = xml_to_sent_dict[ecb_file]
                if topic in train_topics:
                    train_ecb_files_sorted.append((selected_sentences, xml_filename,
                                                   ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_files_sorted.append((selected_sentences, xml_filename,
                                                   ecb_file.replace('.xml', '')))
                else:
                    test_ecb_files_sorted.append((selected_sentences, xml_filename,
                                                  ecb_file.replace('.xml', '')))

        for ecb_file in ecb_plus_files:
            if parse_all or ecb_file in xml_to_sent_dict:
                xml_filename = os.path.join(os.path.join(args.ecb_path,dir),ecb_file)
                if parse_all:
                    selected_sentences = None
                else:
                    selected_sentences = xml_to_sent_dict[ecb_file]
                if topic in train_topics:
                    train_ecb_plus_files_sorted.append((selected_sentences,
                                                        xml_filename, ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_plus_files_sorted.append((selected_sentences,
                                                        xml_filename, ecb_file.replace('.xml', '')))
                else:
                    test_ecb_plus_files_sorted.append(
                        (selected_sentences, xml_filename, ecb_file.replace('.xml', '')))

    train_files = train_ecb_files_sorted + train_ecb_plus_files_sorted
    test_files = test_ecb_files_sorted + test_ecb_plus_files_sorted
    dev_files = dev_ecb_files_sorted + dev_ecb_plus_files_sorted

    train_extracted_mentions = []
    dev_extracted_mentions = []
    test_extracted_mentions = []

    for doc in train_files:
        read_ecb_plus_doc(doc[0], doc[1],doc[2],train_out,train_extracted_mentions, parse_all, load_singletons)

    for doc in dev_files:
        read_ecb_plus_doc(doc[0], doc[1], doc[2], dev_out, dev_extracted_mentions, parse_all, load_singletons)

    for doc in test_files:
        read_ecb_plus_doc(doc[0], doc[1],doc[2],test_out,test_extracted_mentions, parse_all, load_singletons)

    train_out.close()
    dev_out.close()
    test_out.close()

    save_gold_mention_statistics(train_extracted_mentions, dev_extracted_mentions,
                                 test_extracted_mentions)

    save_split_mentions_to_json('Train', train_extracted_mentions)
    save_split_mentions_to_json('Dev', dev_extracted_mentions)
    save_split_mentions_to_json('Test', test_extracted_mentions)

    all_mentions = train_extracted_mentions + dev_extracted_mentions + test_extracted_mentions

    save_split_mentions_to_json('All', all_mentions)


def main():
    """
        This script processes the ECB+ XML files and saves for each data split (train/dev/test):
        1) A json file contains its mention objects.
        2) text file contains its sentences.
        .
        Runs data processing scripts to turn raw data from (../raw) into
        intermediate data (mention objects and sentences' text) ready for feature extraction
        (saved in ../intermid).
    """
    logger.info('Read ECB+ files')
    if args.data_setup == 1:  # Reads the full ECB+ corpus without singletons (Yang setup)
        parse_selected_sentences(xml_to_sent_dict={}, parse_all=True, load_singletons=False, data_setup=1)
    elif args.data_setup == 2:  # Reads the a reviewed subset of the ECB+ (Cybulska setup)
        xml_to_sent_dict = read_selected_sentences(args.selected_sentences_file)
        parse_selected_sentences(xml_to_sent_dict=xml_to_sent_dict,parse_all=False,
                                 load_singletons=True,data_setup=2)
    logger.info('ECB+ Reading was done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()
