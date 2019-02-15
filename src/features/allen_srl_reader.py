import json

'''
Classes for reading the output of Allen NLP SRL system
'''
class SRLSentence(object):
    '''
    A class represents a sentence in ECB+ and contains a list of predicates (with their arguments)
    extracted from this sentence.

    '''
    def __init__(self, doc_id, sent_id):
        self.ecb_doc_id = doc_id
        self.ecb_sent_id = sent_id
        self.srl = list()  # Predicates list

    def add_srl_vrb(self, srl_vrb):
        '''
        Adds new predicate to the predicates list
        :param srl_vrb: an SRLVerb object, represents a predicate (along with its arguments)
        :return:
        '''
        self.srl.append(srl_vrb)


class SRLArg(object):
    '''
     A class represents an argument
    '''
    def __init__(self, text, tok_ids):
        self.text = text
        self.ecb_tok_ids = tok_ids


class SRLVerb(object):
    '''
        A class represents a predicate (along with its arguments)
    '''
    def __init__(self):
        self.verb = None
        self.arg0 = None
        self.arg1 = None
        self.arg_tmp = None
        self.arg_loc = None
        self.arg_neg = None


def read_srl(file_path):
    '''
    This function gets a json file that contains the output from Allen NLP SRL system, and
    parses it.
    :param file_path: a json file that contains the output from Allen NLP SRL system
    :return: a dictionary contains the SRLSentence objects
    '''
    with open(file_path) as f:
        data = json.load(f)

    all_doc_sentences = {}
    for data_obj in data:
        doc_id = data_obj['ecb_doc_id'].replace('.xml','')  # a string

        if doc_id not in all_doc_sentences:
            all_doc_sentences[doc_id] = {}

        sent_id = data_obj['ecb_sent_id'] # an integer

        if sent_id not in all_doc_sentences[doc_id]:
            all_doc_sentences[doc_id][sent_id] = None

        srl_obj = data_obj['srl']  # a list

        srl_sentences = SRLSentence(doc_id, sent_id)
        for obj in srl_obj:
            srl_verb_obj = SRLVerb()
            if 'verb' in obj and obj['verb'] is not None:
                verb = obj['verb']
                srl_verb_obj.verb = SRLArg(verb['text'], verb['ecb_tok_ids'])
            if 'arg0' in obj and obj['arg0'] is not None:
                arg0 = obj['arg0']
                srl_verb_obj.arg0 = SRLArg(arg0['text'], arg0['ecb_tok_ids'])
            if 'arg1' in obj and obj['arg1'] is not None:
                arg1 = obj['arg1']
                srl_verb_obj.arg1 = SRLArg(arg1['text'], arg1['ecb_tok_ids'])
            if 'arg_tmp' in obj and obj['arg_tmp'] is not None:
                arg_tmp = obj['arg_tmp']
                srl_verb_obj.arg_tmp = SRLArg(arg_tmp['text'], arg_tmp['ecb_tok_ids'])
            if 'arg_loc' in obj and obj['arg_loc'] is not None:
                arg_loc = obj['arg_loc']
                srl_verb_obj.arg_loc = SRLArg(arg_loc['text'], arg_loc['ecb_tok_ids'])
            if 'arg_neg' in obj and obj['arg_neg'] is not None:
                arg_neg = obj['arg_neg']
                srl_verb_obj.arg_neg = SRLArg(arg_neg['text'], arg_neg['ecb_tok_ids'])

            srl_sentences.add_srl_vrb(srl_verb_obj)

        all_doc_sentences[doc_id][sent_id] = srl_sentences

    return all_doc_sentences

