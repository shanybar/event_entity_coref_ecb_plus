
import os
import sys
import _pickle as cPickle
import logging
import argparse

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

parser = argparse.ArgumentParser(description='Reading an ECB+ split  '
                                             'and analyzing the feature extraction performance')

parser.add_argument('--data_path', type=str,
                    help=' The path to the ECB+ processed split file (stored in ../processed)')
parser.add_argument('--output_dir', type=str,
                    help='Directory of output files')
parser.add_argument('--split_name', type=str,
                    help='The split name - e.g. train/dev/test (used for naming output files)')


args = parser.parse_args()

non_coref_args_count = 0
coref_args_count = 0
checked_args_count = 0
file_obj = open(os.path.join(args.output_dir,'{}_args_manual_analysis.txt'.format(args.split_name)),'w')


def is_args_coref(arg_i, arg_j, topic):
    '''
    Checks whether two arguments (i.e. entity mentions) share the same coreference chain
    :param arg_i: the first argument
    :param arg_j: the second argument
    :param topic: their topic object
    :return: True if the two arguments share the same coreference chain and false otherwise.
    '''
    global non_coref_args_count, checked_args_count
    cluster_i = topic.entity_mention_id_to_gold[arg_i]
    cluster_j = topic.entity_mention_id_to_gold[arg_j]
    checked_args_count += 1
    if cluster_i == cluster_j:
        return True
    else:
        non_coref_args_count += 1
    return False


def have_coref_arg(event_i, event_j, topic):
    '''
    Checks whether two events mentions have coreferring arguments.
    The functions also writes to file the non-coreferring arguemnts that have been found
    (for a manual analysis)
    :param arg_i: the first event mention
    :param arg_j: the second event mention
    :param topic: their topic object
    :return: True if the two event mentions have coreferring arguments false otherwise.
    '''
    global file_obj, coref_args_count
    answer = False
    if event_i.arg0 is not None and event_j.arg0 is not None:
        event_i_arg0 = event_i.arg0[1]
        event_j_arg0 = event_j.arg0[1]
        answer = is_args_coref(event_i_arg0, event_j_arg0, topic)
        if answer:
            coref_args_count += 1
            answer =  True
        else:
            file_obj.write('Non-coref arg0: event 1: {}, event 2: {}, a0 1: {} a0 2: {} \n'.format(
                event_i.mention_str, event_j.mention_str, event_i.arg0[0], event_j.arg0[0]
            ))
            file_obj.write('sent 1 ({}, {}): {}\n'.format(event_i.doc_id,event_i.sent_id,topic.docs[event_i.doc_id].sentences[event_i.sent_id].get_raw_sentence()))
            file_obj.write('sent 2 ({}, {}): {}\n\n'.format(event_j.doc_id,event_j.sent_id,topic.docs[event_j.doc_id].sentences[event_j.sent_id].get_raw_sentence()))
    if event_i.arg1 is not None and event_j.arg1 is not None:
        event_i_arg1 = event_i.arg1[1]
        event_j_arg1 = event_j.arg1[1]
        answer = is_args_coref(event_i_arg1, event_j_arg1, topic)
        if answer:
            coref_args_count += 1
            answer =  True
        else:
            file_obj.write('Non-coref arg1: event 1: {}, event 2: {}, a1 1: {} a1 2: {} \n'.format(
                event_i.mention_str, event_j.mention_str, event_i.arg1[0], event_j.arg1[0]
            ))
            file_obj.write('sent 1 ({}, {}): {}\n'.format(event_i.doc_id,event_i.sent_id,topic.docs[event_i.doc_id].sentences[event_i.sent_id].get_raw_sentence()))
            file_obj.write('sent 2 ({}, {}): {}\n\n'.format(event_j.doc_id,event_j.sent_id,topic.docs[event_j.doc_id].sentences[event_j.sent_id].get_raw_sentence()))
    if event_i.amtmp is not None and event_j.amtmp is not None:
        event_i_amtmp = event_i.amtmp[1]
        event_j_amtmp = event_j.amtmp[1]
        answer = is_args_coref(event_i_amtmp, event_j_amtmp, topic)
        if answer:
            coref_args_count += 1
            answer =  True
        else:
            file_obj.write('Non-coref amtmp: event 1: {}, event 2: {}, amtmp 1: {} amtmp 2: {} \n'.format(
                event_i.mention_str, event_j.mention_str, event_i.amtmp[0], event_j.amtmp[0]
            ))
            file_obj.write('sent 1 ({}, {}): {}\n'.format(event_i.doc_id,event_i.sent_id,topic.docs[event_i.doc_id].sentences[event_i.sent_id].get_raw_sentence()))
            file_obj.write('sent 2 ({}, {}): {}\n\n'.format(event_j.doc_id,event_j.sent_id,topic.docs[event_j.doc_id].sentences[event_j.sent_id].get_raw_sentence()))
    if event_i.amloc is not None and event_j.amloc is not None:
        event_i_amloc = event_i.amloc[1]
        event_j_amloc = event_j.amloc[1]
        answer = is_args_coref(event_i_amloc, event_j_amloc, topic)
        if answer:
            coref_args_count += 1
            answer = True
        else:
            file_obj.write('Non-coref amloc: event 1: {}, event 2: {}, amloc 1: {} amloc 2: {} \n'.format(
                event_i.mention_str, event_j.mention_str, event_i.amloc[0], event_j.amloc[0]
            ))
            file_obj.write('sent 1 ({}, {}): {}\n'.format(event_i.doc_id,event_i.sent_id,topic.docs[event_i.doc_id].sentences[event_i.sent_id].get_raw_sentence()))
            file_obj.write('sent 2 ({}, {}): {}\n\n'.format(event_j.doc_id,event_j.sent_id,topic.docs[event_j.doc_id].sentences[event_j.sent_id].get_raw_sentence()))
    return answer


def have_args(event):
    '''
     Checks whether an event mention has any extracted arguments
    :param event: an event mention object
    :return: True, if the event mention has any extracted arguments, and false otherwise.
    '''
    if event.arg0 is not None or event.arg1 is not None or event.amloc is not None or event.amtmp is not None:
        return True
    return False


def have_double_args(event):
    '''
    Check whether an event mention has an entity mention that fills different
    semantic roles (due to errors in feature extraction)
    :param event: an event mention
    :return: True, if the event mention has an entity mention that fills different
    semantic roles, and false otherwise.
    '''
    if event.arg0 is not None or event.arg1 is not None:
        if event.arg0 == event.arg1:
            return True
    if event.arg0 is not None and event.amloc is not None:
        if event.arg0 == event.amloc:
            return True
    if event.arg0 is not None and event.amtmp is not None:
        if event.arg0 == event.amtmp:
            return True
    if event.arg1 is not None and event.amloc is not None:
        if event.arg1 == event.amloc:
            return True
    if event.arg1 is not None and event.amtmp is not None:
        if event.arg1 == event.amtmp:
            return True
    if event.amloc is not None and event.amtmp is not None:
        if event.amloc == event.amtmp:
            return True
    return False


def topic_to_mention_list(topic, is_gold):
    '''
    Gets a Topic object and extracts its event/entity mentions
    :param topic: a Topic object
    :param is_gold: a flag that denotes whether to extract gold mention or predicted mentions
    :return: list of the topic's mentions
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


def main():
    '''
    The script reads an ECB+ split and analyzes the feature extraction performance by counting
    how many event mention have extracted arguments and how many arguments of the same semantic role
    are coreferring among event mentions that share the same gold coreference chain.

    Note that this script should be used after running feature extraction.
    '''

    global non_coref_args_count, checked_args_count, coref_args_count
    non_coref_args_count = 0
    checked_args_count = 0
    coref_args_count = 0
    logger.info('Loading data for analysis...')
    with open(args.data_path, 'rb') as f:
        data = cPickle.load(f)

    events_with_coref_arg = 0
    checked_pairs = 0
    event_mentions_count = 0
    events_with_args = 0
    events_with_double_args = 0
    coref_events_one_coref_arg_link = 0 #  how many event mentions (in a gold cluster) have a coref argument with at least one event mention in their cluster
    for topic_id, topic in data.topics.items():

        event_mentions, entity_mentions = topic_to_mention_list(topic, is_gold=True)
        pred_event_mentions, pred_entity_mentions = topic_to_mention_list(topic, is_gold=False)
        with open(os.path.join(args.output_dir,
                               '{}_mentions_per_topic.txt'.format(args.split_name)), 'a') as f:
            f.write('Topic id {} \n'.format(topic_id))
            f.write('Number of gold event mentions - {}\n'.format(len(event_mentions)))
            f.write('Number of gold entity mentions - {}\n'.format(len(entity_mentions)))
            if len(pred_event_mentions) > 0:
                f.write('Number of predicted event mentions - {}\n'.format(len(pred_event_mentions)))
            if len(pred_entity_mentions) > 0:
                f.write('Number of predicted entity mentions - {}\n'.format(len(pred_entity_mentions)))
        topic.event_mentions = event_mentions
        topic.entity_mentions = entity_mentions
        topic.set_gold_clusters()
        event_mentions_count += len(event_mentions)

        event_clusters = topic.gold_event_clusters
        for cluster in event_clusters:
            cluster_mentions = cluster

            for event in cluster_mentions:
                if have_args(event):
                    events_with_args += 1
                if have_double_args(event):
                    events_with_double_args += 1

            for i in range(len(cluster_mentions)-1):
                have_one_coref_arg_link = False
                for j in range(i+1,len(cluster_mentions)):
                    event_i = cluster_mentions[i]
                    event_j = cluster_mentions[j]
                    if have_coref_arg(event_i, event_j, topic):
                        events_with_coref_arg += 1
                        if not have_one_coref_arg_link:
                            have_one_coref_arg_link = True
                            coref_events_one_coref_arg_link += 1
                    checked_pairs += 1

    coref_percent = (events_with_coref_arg/float(checked_pairs)) * 100.0
    non_coref_percent = (non_coref_args_count/float(checked_args_count)) * 100.0
    coref_args_percent = (coref_args_count / float(checked_args_count)) * 100.0

    with open(os.path.join(args.output_dir,
                           '{}_predicate_args_extraction_stats.txt'.format(args.split_name)), 'w') as f:
        f.write(' Event mentions - {} \n'.format(event_mentions_count))
        f.write('{} % of event mentions have at least one argument.\n'.format(100.0 * (events_with_args/float(event_mentions_count))))
        f.write('{} % of coreferential event mentions pairs have at least one coreferential argument.\n'.format(coref_percent))
        f.write('Number of coreferential arguments - {}\n'.format(coref_args_count))
        f.write('Number of non-coreferential arguments - {}\n'.format(non_coref_args_count))
        f.write('{} % of coreferential arguments.\n'.format(coref_args_percent))
        f.write('{} % of non-coreferential arguments.\n'.format(non_coref_percent))
        f.write('{} % of event mentions with double args.\n'.format(
            100.0 * (events_with_double_args / float(event_mentions_count))))
        f.write('{} % of event mentions have at least one coref event'
                ' mention with coref argument.\n'.format(
            100.0 * (coref_events_one_coref_arg_link / float(event_mentions_count))))
    file_obj.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()