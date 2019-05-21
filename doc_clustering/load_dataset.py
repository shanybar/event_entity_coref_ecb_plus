# -*- coding: utf-8 -*-
import os
import _pickle as cPickle
import logging
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Parsing ECB+ corpus')

parser.add_argument('--ecb_path', type=str,
                    help=' The path to the ECB+ corpus')
parser.add_argument('--output_dir', type=str,
                        help=' The directory of the output files')

args = parser.parse_args()


class Token(object):
    def __init__(self, text, sent_id, tok_id, rel_id=None):
        '''

        :param text: The token text
        :param sent_id: The sentence id
        :param tok_id: The token id
        :param rel_id: The relation id
        '''

        self.text = text
        self.sent_id = sent_id
        self.tok_id = tok_id
        self.rel_id = rel_id


def load_ecb_plus_raw_doc(doc_filename, doc_id):
    ecb_file = open(doc_filename, 'r')
    tree = ET.parse(ecb_file)
    root = tree.getroot()

    tokens = []
    for token in root.findall('token'):
        if 'plus' in doc_id and int(token.attrib['sentence']) == 0:
            continue
        tokens.append(token.text)

    return ' '.join(tokens)


def load_raw_test_data():
    test_topics = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    dirs = os.listdir(args.ecb_path)
    docs = {}

    for dir in dirs:
        doc_files = os.listdir(os.path.join(args.ecb_path, dir))
        for doc_file in doc_files:
            doc_id = doc_file.replace('.xml', '')
            if int(doc_id.split('_')[0]) in test_topics:
                xml_filename = os.path.join(os.path.join(args.ecb_path, dir),doc_file)
                raw_doc = load_ecb_plus_raw_doc(xml_filename, doc_id)
                docs[doc_id] = raw_doc

    return docs


def main():
    test_docs = load_raw_test_data()
    with open(os.path.join(args.output_dir, 'test_raw_docs'), 'wb') as f:
        cPickle.dump(test_docs, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()
