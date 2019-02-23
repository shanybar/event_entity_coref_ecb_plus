import os
import json
import logging
import argparse
'''
Convert predicted mention file to JSON file
'''

from mention_data import MentionData

parser = argparse.ArgumentParser(description='Parsing ECB+ corpus')

parser.add_argument('--test_predicted_file', type=str,
                    help=' The path to a file contains the predicted mentions extracted'
                         ' from the test set')
parser.add_argument('--output_dir', type=str,
                        help=' The directory of the output files')

args = parser.parse_args()


def obj_dict(obj):
    return obj.__dict__


def calc_split_statistics(dataset_split):
    '''
    This function calculates and saves the statistics of a split (train/dev/test) into a file.
    :param dataset_split: a list that contains all the mention objects in the split
    '''
    event_mentions_count = 0
    human_mentions_count = 0
    non_human_mentions_count = 0
    loc_mentions_count = 0
    time_mentions_count = 0


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

    statistics_file_name = os.path.join(args.output_dir, 'test_predicted_mention_stats.txt')

    with open(statistics_file_name, 'w') as f:

        f.write('{} statistics\n'.format('test'))
        f.write('-------------------------\n')
        f.write( 'Number of event mentions - {}\n'.format(event_mentions_count))
        f.write( 'Number of human participants mentions - {}\n'.format(human_mentions_count))
        f.write( 'Number of non-human participants mentions - {}\n'.format(non_human_mentions_count))
        f.write( 'Number of location mentions - {}\n'.format(loc_mentions_count))
        f.write( 'Number of time mentions - {}\n'.format(time_mentions_count))
        f.write( 'Total number of mentions - {}\n'.format(len(dataset_split)))

        f.write('\n')


def load_predicted_mentions():
    doc_changed = False
    sent_changed = False
    last_doc_name = None
    last_sent_id = None

    extracted_mentions = []
    mention_tokens = []
    mention_token_numbers = []
    prev_mention_bio = 'O'
    for line in open(args.test_predicted_file, 'r'):
        stripped_line = line.strip()
        # try:
        if stripped_line:
            doc_id, sent_id, token_num, token, _, curr_pred_mention_bio =\
                stripped_line.split('\t')
            doc_id = doc_id.replace('.xml', '')
        # except:
        #     row = stripped_line.split('\t')
        #     clean_row = []
        #     for item in row:
        #         if item:
        #             clean_row.append(item)
        #     doc_id, sent_id, token_num, word, coref_chain = clean_row
        #     doc_id = doc_id.replace('.xml', '')

        if stripped_line:
            sent_id = int(sent_id)

            if last_doc_name is None:
                last_doc_name = doc_id
            elif last_doc_name != doc_id:
                doc_changed = True
                sent_changed = True

            if last_sent_id is None:
                last_sent_id = sent_id
            elif last_sent_id != sent_id:
                sent_changed = True
            if sent_changed:
                # check if there is a previous mention (and create a new MentionData instance, if needed)
                if prev_mention_bio.startswith('B-') or prev_mention_bio.startswith('I-'):
                    if doc_changed:
                        mention_doc_id = last_doc_name
                        # last_doc_name = doc_id
                        # doc_changed = False
                    else:
                        mention_doc_id = doc_id

                    clean_prev_bio = prev_mention_bio.replace('B-', '').replace('I-', '')
                    mention_obj = MentionData(mention_doc_id, last_sent_id, mention_token_numbers,
                                              ' '.join(mention_tokens),
                                              '-', clean_prev_bio, is_continuous=True,
                                              is_singleton=False, score=float(-1))

                    extracted_mentions.append(mention_obj)

                if doc_changed:
                    doc_changed = False
                    last_doc_name = doc_id

                sent_changed = False
                last_sent_id = sent_id
                prev_mention_bio = 'O'

            if curr_pred_mention_bio.startswith('B-'):
                if prev_mention_bio == 'O':
                    # Start a new mention, there is no previous mention
                    mention_tokens = [token]
                    mention_token_numbers = [int(token_num)]

                elif prev_mention_bio.startswith('B-') or prev_mention_bio.startswith('I-'):
                    # take care of the previous mention (create a new MentionData instance)
                    clean_prev_bio = prev_mention_bio.replace('B-','').replace('I-','')
                    mention_obj = MentionData(doc_id, sent_id, mention_token_numbers,
                                              ' '.join(mention_tokens),
                                              '-', clean_prev_bio, is_continuous=True,
                                              is_singleton=False, score=float(-1))

                    extracted_mentions.append(mention_obj)

                    # Start a new mention
                    mention_tokens = [token]
                    mention_token_numbers = [int(token_num)]

                prev_mention_bio = curr_pred_mention_bio

            elif curr_pred_mention_bio.startswith('I-'):
                mention_tokens.append(token)
                mention_token_numbers.append(int(token_num))
                prev_mention_bio = curr_pred_mention_bio
            elif curr_pred_mention_bio == 'O':
                if prev_mention_bio.startswith('B-') or prev_mention_bio.startswith('I-'):
                    # take care of the previous mention (create a new MentionData instance)
                    clean_prev_bio = prev_mention_bio.replace('B-', '').replace('I-', '')
                    mention_obj = MentionData(doc_id, sent_id, mention_token_numbers,
                                              ' '.join(mention_tokens),
                                              '-', clean_prev_bio, is_continuous=True,
                                              is_singleton=False, score=float(-1))

                    extracted_mentions.append(mention_obj)

                prev_mention_bio = curr_pred_mention_bio

    calc_split_statistics(extracted_mentions)

    event_mentions = []
    entity_mentions = []

    for mention_obj in extracted_mentions:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type:
            event_mentions.append(mention_obj)
        else:
            entity_mentions.append(mention_obj)

    json_event_filename = os.path.join(args.output_dir, 'ECB_Test_Event_pred_mentions.json')
    json_entity_filename = os.path.join(args.output_dir, 'ECB_Test_Entity_pred_mentions.json')
    with open(json_event_filename, 'w') as f:
        json.dump(event_mentions, f, default=obj_dict, indent=4, sort_keys=True)

    with open(json_entity_filename, 'w') as f:
        json.dump(entity_mentions, f, default=obj_dict, indent=4, sort_keys=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info('Read predicted mentions file...')
    load_predicted_mentions()