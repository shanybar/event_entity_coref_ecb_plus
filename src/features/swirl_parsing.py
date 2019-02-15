import os
import sys
for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

from classes import *


def parse_swirl_sent(sent_id, sent_tokens):
    '''
    This function gets a sentence in a SwiRL "format" and extracts the predicates
    and their arguments from it.
    The function returns a dictionary in the following structure:
    dict[key3] = Srl_info object
    while key3 is a token id of an extracted event.
    See a documentation about Srl_info object in classed.py.
    :param sent_id: the sentence ordinal number in the document
    :param sent_tokens: the sentence's tokens
    :return: a dictionary as mentioned above
    '''
    col = 0
    event_dict = {}
    for tok_idx, tok in enumerate(sent_tokens):
        if tok[0] != '-':
            col += 1
            events_args = {}
            # look for the arguments
            for arg_idx, arg in enumerate(sent_tokens):
                if '(' in arg[col] and ')' in arg[col]:
                    # one word argument
                    arg_name = arg[col][1:-1]
                    arg_name = arg_name.replace('*','')
                    arg_name = arg_name.replace('R-', '')
                    events_args[arg_name] = [arg_idx]
                elif '(' in arg[col]:
                    # argument with two or more words
                    arg_bound_found = False
                    arg_name = arg[col][1:-1]
                    arg_name = arg_name.replace('*', '')
                    arg_name = arg_name.replace('R-', '')
                    events_args[arg_name] = [arg_idx]
                    bound_idx = arg_idx + 1
                    while bound_idx < len(sent_tokens) and not arg_bound_found:
                        if ')' in sent_tokens[bound_idx][col]:
                            events_args[arg_name].append(bound_idx)
                            arg_bound_found = True
                        bound_idx += 1
            # save the arguments per predicate
            event_dict[tok_idx] = Srl_info(sent_id, events_args,tok_idx, tok[0])
    return event_dict


def parse_swirl_file(fname, file_path,srl_data):
    '''
    This function gets the path to the output files of SwiRL,
    extracts the predicates and their arguments for each sentence in each document
    and returns a dictionary in the following structure:
    dict[key1][key2] = dict[key3].
    dict[key3] contains a Srl_info object.
    key1 - document id
    key2 - sent id
    key3 - token id of an extracted event
    :param fname: SwiRL output file to parse
    :param file_path: path to SwiRL folder
    :param srl_data: the dictionary
    :return: a dictionary as mentioned above
    '''
    swirl_file = open(file_path, 'r')
    splitted_fname = fname.split('.')
    srl_data[splitted_fname[0]] = {}
    sent_id = 0
    sent_tokens = []
    for line in swirl_file:
        temp_line = line.strip().split()
        if not temp_line:
            srl_data[splitted_fname[0]][sent_id] = parse_swirl_sent(sent_id, sent_tokens)
            sent_id += 1
            sent_tokens = []
        else:
            sent_tokens.append(temp_line)
    # parse the last sentence
    srl_data[splitted_fname[0]][sent_id] = parse_swirl_sent(sent_id, sent_tokens)
    swirl_file.close()


def parse_swirl_output(srl_files_path):
    '''
    This function gets the path to the output files of SwiRL and parse
    each output file
    :param srl_files_path: the path to the output files of SwiRL
    :return: a dictionary (see the previous function's documentation)
    '''
    srl_data = {}
    for root, directory, files in os.walk(srl_files_path):
        for f in files:
            splitted = f.split('.')
            fname = splitted[1] + '.' + splitted[2]
            parse_swirl_file(fname, os.path.join(root, f),srl_data)

    return srl_data


