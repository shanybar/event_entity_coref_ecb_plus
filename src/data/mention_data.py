
class MentionData(object):
    '''
    An helper class for a mid-representation of a mention when reading the corpus.
    '''
    def __init__(self, doc_id, sent_id, tokens_numbers, tokens_str, coref_chain, mention_type='ACT',
                 is_continuous=True, is_singleton=False, score=float(-1)):
        '''

        :param doc_id: the mention's document ID
        :param sent_id: the mention's sentence ID
        :param tokens_numbers: the token IDs in the mention's text span
        :param tokens_str: the mention's string
        :param coref_chain: the mention's gold coreference chain
        :param mention_type: the mention's type, marked with ACT for event mention and with the
        following types for entity mentions: HUM for human participant, NON for non-human participant,
        LOC for location, and TIM for time.
        :param is_continuous: a variable indicates whether the mention span is continuous or not
        :param is_singleton: a variable indicates whether the mention belongs to a singleton  coreference
        cluster ( a cluster which contains only a single mention)
        :param score: a confidence score of the span for being a real mention (this score relates to
        predicted mentions only, and it sets to -1 dealing with gold mentions)
        '''
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.tokens_number = tokens_numbers
        self.tokens_str = tokens_str
        self.mention_type = mention_type
        self.coref_chain = coref_chain
        self.is_continuous = is_continuous
        self.is_singleton = is_singleton
        self.score = score
