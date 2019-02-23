import os
import gc
import sys
import json
import random
import subprocess
import numpy as np

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

import _pickle as cPickle
import logging
import argparse


parser = argparse.ArgumentParser(description='Testing the regressors')

parser.add_argument('--config_path', type=str,
                    help=' The path configuration json file')
parser.add_argument('--out_dir', type=str,
                    help=' The directory to the output folder')

args = parser.parse_args()

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logging.basicConfig(filename=os.path.join(args.out_dir, "test_log.txt"),
                    level=logging.INFO, filemode="w")

# Loads a json configuration file (test_config.json)
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

# Saves a json configuration file (test_config.json) in the experiment folder
with open(os.path.join(args.out_dir,'test_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

random.seed(config_dict["random_seed"])
np.random.seed(config_dict["random_seed"])

if config_dict["gpu_num"] != -1:
    os.environ["CUDA_VISIBLE_DEVICES"]= str(config_dict["gpu_num"])
    args.use_cuda = True
else:
    args.use_cuda = False

import torch

args.use_cuda = args.use_cuda and torch.cuda.is_available()

torch.manual_seed(config_dict["seed"])
if args.use_cuda:
    torch.cuda.manual_seed(config_dict["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Testing with CUDA')

from scorer import *
from classes import *
from eval_utils import *
from model_utils import *

def read_conll_f1(filename):
    '''
    This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
    B-cubed and the CEAF-e and calculates CoNLL F1 score.
    :param filename: a file stores the scorer's results.
    :return: the CoNLL F1
    '''
    f1_list = []
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                f1_list.append(float(new_line.split(': ')[-1][:-1]))

    muc_f1 = f1_list[1]
    bcued_f1 = f1_list[3]
    ceafe_f1 = f1_list[7]

    return (muc_f1 + bcued_f1 + ceafe_f1)/float(3)


def run_conll_scorer():
    if config_dict["test_use_gold_mentions"]:
        event_response_filename = os.path.join(args.out_dir, 'CD_test_event_mention_based.response_conll')
        entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_mention_based.response_conll')
    else:
        event_response_filename = os.path.join(args.out_dir, 'CD_test_event_span_based.response_conll')
        entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_span_based.response_conll')

    event_conll_file = os.path.join(args.out_dir,'event_scorer_cd_out.txt')
    entity_conll_file = os.path.join(args.out_dir,'entity_scorer_cd_out.txt')

    event_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
            (config_dict["event_gold_file_path"], event_response_filename, event_conll_file))

    entity_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
            (config_dict["entity_gold_file_path"], entity_response_filename, entity_conll_file))

    processes = []
    print('Run scorer command for cross-document event coreference')
    processes.append(subprocess.Popen(event_scorer_command, shell=True))

    print('Run scorer command for cross-document entity coreference')
    processes.append(subprocess.Popen(entity_scorer_command, shell=True))

    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)

    print ('Running scorers has been done.')
    print ('Save results...')

    scores_file = open(os.path.join(args.out_dir, 'conll_f1_scores.txt'), 'w')

    event_f1 = read_conll_f1(event_conll_file)
    entity_f1 = read_conll_f1(entity_conll_file)
    scores_file.write('Event CoNLL F1: {}\n'.format(event_f1))
    scores_file.write('Entity CoNLL F1: {}\n'.format(entity_f1))

    scores_file.close()


def test_model(test_set):
    '''
    Loads trained event and entity models and test them on the test set
    :param test_set: a Corpus object, represents the test split
    '''
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    cd_event_model = load_check_point(config_dict["cd_event_model_path"])
    cd_entity_model = load_check_point(config_dict["cd_entity_model_path"])

    cd_event_model.to(device)
    cd_entity_model.to(device)

    doc_to_entity_mentions = load_entity_wd_clusters(config_dict)

    _,_ = test_models(test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)

    run_conll_scorer()


def main():
    '''
    This script loads the trained event and entity models and test them on the test set
    '''
    print('Loading test data...')
    logging.info('Loading test data...')
    with open(config_dict["test_path"], 'rb') as f:
        test_data = cPickle.load(f)

    print('Test data have been loaded.')
    logging.info('Test data have been loaded.')

    test_model(test_data)


if __name__ == '__main__':

    main()