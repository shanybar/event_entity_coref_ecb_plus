import os
import sys
import _pickle as cPickle
import logging
import argparse

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

from eval_utils import *

parser = argparse.ArgumentParser(description='Reads the ECB+ test split  '
                                             'and creates gold files')

parser.add_argument('--test_path', type=str,
                    help=' The path to the ECB+ processed test data (stored in ../processed)')
parser.add_argument('--gold_files_dir', type=str,
                    help=' Where to locate the gold files')
parser.add_argument('--mention_based_key_file', action='store_true',
                    help='')

args = parser.parse_args()

out_dir = args.gold_files_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def main():
    '''
    This script reads the test split after feature extraction has been done and creates
    gold (key) files for CoNLL scorer for both event and entity mentions.
    '''
    logger.info('Loading test data...')
    with open(args.test_path, 'rb') as f:
        test_data = cPickle.load(f)

    logger.info('Test data have been loaded.')

    if not args.mention_based_key_file:
        logger.info('Creating span-based event mentions key file')
        out_file = os.path.join(args.gold_files_dir, 'CD_test_event_span_based.key_conll')
        write_span_based_cd_coref_clusters(test_data, out_file, is_event=True, is_gold=True,
                                           use_gold_mentions=True)
        logger.info('Creating span-based entity mentions key file')
        out_file = os.path.join(args.gold_files_dir, 'CD_test_entity_span_based.key_conll')
        write_span_based_cd_coref_clusters(test_data, out_file, is_event=False, is_gold=True,
                                           use_gold_mentions=True)

    else:
        logger.info('Creating mention-based event mentions key file')
        out_file = os.path.join(args.gold_files_dir, 'CD_test_event_mention_based.key_conll')
        write_mention_based_cd_clusters(test_data, is_event=True, is_gold=True, out_file=out_file)

        out_file = os.path.join(args.gold_files_dir, 'WD_test_event_mention_based.key_conll')
        write_mention_based_wd_clusters(test_data, is_event=True, is_gold=True, out_file=out_file)

        logger.info('Creating mention-based entity mentions key file')
        out_file = os.path.join(args.gold_files_dir, 'CD_test_entity_mention_based.key_conll')
        write_mention_based_cd_clusters(test_data, is_event=False, is_gold=True, out_file=out_file)

        out_file = os.path.join(args.gold_files_dir, 'WD_test_entity_mention_based.key_conll')
        write_mention_based_wd_clusters(test_data, is_event=False, is_gold=True, out_file=out_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()