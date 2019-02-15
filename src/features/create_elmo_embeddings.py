import logging

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder

logger = logging.getLogger(__name__)


class ElmoEmbedding(object):
    '''
    A wrapper class for the ElmoEmbedder of Allen NLP
    '''
    def __init__(self, options_file, weight_file):
        logger.info('Loading Elmo Embedding module')
        self.embedder = ElmoEmbedder(options_file, weight_file)
        logger.info('Elmo Embedding module loaded successfully')

    def get_elmo_avg(self, sentence):
        '''
        This function gets a sentence object and returns and ELMo embeddings of
        each word in the sentences (specifically here, we average over the 3 ELMo layers).
        :param sentence: a sentence object
        :return: the averaged ELMo embeddings of each word in the sentences
        '''
        tokenized_sent = sentence.get_tokens_strings()
        embeddings = self.embedder.embed_sentence(tokenized_sent)
        output = np.average(embeddings, axis=0)

        return output




