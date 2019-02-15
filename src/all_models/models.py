import math
import torch
import torch.nn as nn
from model_utils import *
import torch.nn.functional as F
import torch.autograd as autograd


class CorefScorer(nn.Module):
    '''
    An abstract class represents a coreference pairwise scorer.
    Inherits Pytorch's Module class.
    '''
    def __init__(self, word_embeds, word_to_ix,vocab_size, char_embedding, char_to_ix, char_rep_size
                 , lexical_feats_type, args_feats_type, use_mult, use_diff, feature_size,
                 mention_span_lstm_hidden_size):
        '''
        C'tor for CorefScorer object
        :param word_embeds: pre-trained word embeddings
        :param word_to_ix: a mapping between a word (string) to
        its index in the word embeddings' lookup table
        :param vocab_size:  the vocabulary size
        :param char_embedding: initial character embeddings
        :param char_to_ix:  mapping between a character to
        its index in the character embeddings' lookup table
        :param char_rep_size: hidden size of the character LSTM
        :param lexical_feats_type: string represents the lexical features type (currently unused)
        :param args_feats_type:  string represents the predicate-argument
         features type (currently unused)
        :param use_mult: a boolean indicates whether to use element-wise multiplication in the
        input layer
        :param use_diff: a boolean indicates whether to use element-wise differentiation in the
        input layer
        :param feature_size: embeddings size of binary features
        :param mention_span_lstm_hidden_size: hidden size for the mention-context Bi-LSTM (currently
        replaced by ELMo embeddings)
        '''
        super(CorefScorer, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_embeds.shape[1])

        self.embed.weight.data.copy_(torch.from_numpy(word_embeds))
        self.embed.weight.requires_grad = False # pre-trained word embeddings are fixed
        self.word_to_ix = word_to_ix

        self.char_embeddings = nn.Embedding(len(char_to_ix.keys()), char_embedding.shape[1])
        self.char_embeddings.weight.data.copy_(torch.from_numpy(char_embedding))
        self.char_embeddings.weight.requires_grad = True
        self.char_to_ix = char_to_ix
        self.embedding_dim = word_embeds.shape[1]
        self.char_hidden_dim = char_rep_size
        self.mention_span_lstm_hidden_size = mention_span_lstm_hidden_size

        self.char_lstm = nn.LSTM(input_size=char_embedding.shape[1],hidden_size= self.char_hidden_dim,num_layers=1,
                                 bidirectional=False)

        # This Bi-LSTM is currently unused (replaced by ELMo)
        self.mention_span_lstm = nn.LSTM(input_size=self.embedding_dim + self.char_hidden_dim,
                                         hidden_size=self.mention_span_lstm_hidden_size // 2, num_layers=1,
                                         bidirectional=True)

        # binary features for coreferring arguments/predicates
        self.coref_role_embeds = nn.Embedding(2,feature_size)

        self.lexical_feats_type = lexical_feats_type
        self.args_feats_type = args_feats_type
        self.use_mult = use_mult
        self.use_diff = use_diff
        self.model_type = 'abstract_scorer'

    def init_mention_span_hidden(self, device):
        '''
        initializes hidden states the mention-context Bi-LSTM (currently
        replaced by ELMo embeddings)
        :param device: gpu/cpu Pytorch device
        :return: initialized hidden states (tensors)
        '''
        return (torch.randn((2, 1, self.mention_span_lstm_hidden_size // 2), requires_grad=True).to(device),
                torch.randn((2, 1, self.mention_span_lstm_hidden_size // 2), requires_grad=True).to(device))

    # Unused (replaced by ELMo)
    def get_span_vec(self, embedded_sentence_list, span_start_idx, span_end_idx, device):
        '''
        Runs a bi-directional LSTM on a mention's sentence to embed its span and context,
        returns a concatenation between the output tensor of mention's start index and the
        output tensor of mention's end index
        :param embedded_sentence_list: list of the sentence's pre-trained word embeddings
        :param span_start_idx: the mention span's start index
        :param span_end_idx: the mention span's end index
        :param device: gpu/cpu Pytorch device
        :return: a concatenation between the output tensor of mention's start index and the
        output tensor of mention's end index
        '''
        span_hidden = self.init_mention_span_hidden(device)
        embedded_sentence_tensor = torch.cat(embedded_sentence_list, 0).view(len(embedded_sentence_list), 1, -1)
        span_out, span_hidden = self.mention_span_lstm(embedded_sentence_tensor, span_hidden)
        start_span_out = span_out[span_start_idx, :, :]
        end_span_out = span_out[span_end_idx, :, :]
        span_rep = torch.cat([start_span_out, end_span_out], 1)

        return span_rep

    def init_char_hidden(self, device):
        '''
        initializes hidden states the character LSTM
        :param device: gpu/cpu Pytorch device
        :return: initialized hidden states (tensors)
        '''
        return (torch.randn((1, 1, self.char_hidden_dim ), requires_grad=True).to(device),
                torch.randn((1, 1, self.char_hidden_dim ), requires_grad=True).to(device))

    def get_char_embeds(self, seq, device):
        '''
        Runs a LSTM on a list of character embeddings and returns the last output state
        :param seq: a list of character embeddings
        :param device:  gpu/cpu Pytorch device
        :return: the LSTM's last output state
        '''
        self.char_hidden = self.init_char_hidden(device)
        input_char_seq = self.prepare_chars_seq(seq, device)
        char_embeds = self.char_embeddings(input_char_seq).view(len(seq), 1, -1)
        char_lstm_out, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
        char_vec = char_lstm_out[-1]

        return char_vec

    def prepare_chars_seq(self, seq, device):
        '''
        Given a string represents a word or a phrase, this method converts the sequence
        to a list of character embeddings
        :param seq: a string represents a word or a phrase
        :param device: device:  gpu/cpu Pytorch device
        :return: a list of character embeddings
        '''
        idxs = []
        for w in seq:
            if w in self.char_to_ix:
                idxs.append(self.char_to_ix[w])
            else:
                lower_w = w.lower()
                if lower_w in self.char_to_ix:
                    idxs.append(self.char_to_ix[lower_w])
                else:
                    idxs.append(self.char_to_ix['<UNK>'])
                    print('can find char {}'.format(w))
        tensor = torch.tensor(idxs,dtype=torch.long).to(device)

        return tensor

    def forward(self, clusters_pairs_tensor):
        '''
        Unimplemented abstract method
        :param clusters_pairs_tensor:
        :return:
        '''
        return


class CDCorefScorer(CorefScorer):
    '''
    A class inherits CorefScorer class, representing a cross-document (CD)
    pairwise coreference scorer
    '''
    def __init__(self, word_embeds, word_to_ix, vocab_size, dims, char_embedding, char_to_ix, char_rep_size,
                 lexical_feats_type, args_feats_type,use_mult, use_diff, feature_size,
                 mention_span_lstm_hidden_size):
        '''
        C'tor for CDCorefScorer object
        :param word_embeds: pre-trained word embeddings
        :param word_to_ix: a mapping between a word (string) to
        its index in the word embeddings' lookup table
        :param vocab_size:  the vocabulary size
        :param char_embedding: initial character embeddings
        :param char_to_ix:  mapping between a character to
        its index in the character embeddings' lookup table
        :param char_rep_size: hidden size of the character LSTM
        :param lexical_feats_type: string represents the lexical features type (currently unused)
        :param args_feats_type:  string represents the predicate-argument
         features type (currently unused)
        :param use_mult: a boolean indicates whether to use element-wise multiplication in the
        input layer
        :param use_diff: a boolean indicates whether to use element-wise differentiation in the
        input layer
        :param feature_size: embeddings size of binary features
        :param mention_span_lstm_hidden_size: hidden size for the mention-context Bi-LSTM (currently
        replaced by ELMo embeddings)
        '''
        super(CDCorefScorer, self).__init__(word_embeds, word_to_ix, vocab_size, char_embedding,
                                            char_to_ix, char_rep_size, lexical_feats_type,
                                            args_feats_type, use_mult, use_diff, feature_size,
                                            mention_span_lstm_hidden_size)

        self.input_dim = dims[0]
        self.hidden_dim_1 = dims[1]
        self.hidden_dim_2 = dims[2]
        self.out_dim = 1

        self.hidden_layer_1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.hidden_layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.out_layer = nn.Linear(self.hidden_dim_2, self.out_dim)

        self.model_type = 'CD_scorer'

    def forward(self, clusters_pair_tensor):
        '''
        The forward method - pass the input tensor through a feed-forward neural network
        :param clusters_pair_tensor: an input tensor consists of a concatenation between
        two mention representations, their element-wise multiplication and a vector of binary features
        (each feature embedded as 50 dimensional embeddings)
        :return: a predicted confidence score (between 0 to 1) of the mention pair to be in the
        same coreference chain (aka cluster).
        '''
        first_hidden = F.relu(self.hidden_layer_1(clusters_pair_tensor))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = F.sigmoid(self.out_layer(second_hidden))

        return out

