import math
import torch
import torch.nn as nn
from model_utils import *
import torch.nn.functional as F
import torch.autograd as autograd


class CDCorefScorer(nn.Module):
    '''
    An abstract class represents a coreference pairwise scorer.
    Inherits Pytorch's Module class.
    '''
    def __init__(self, word_embeds, word_to_ix,vocab_size, char_embedding, char_to_ix, char_rep_size
                 , dims, use_mult, use_diff, feature_size):
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
        :param dims: list holds the layer dimensions
        :param use_mult: a boolean indicates whether to use element-wise multiplication in the
        input layer
        :param use_diff: a boolean indicates whether to use element-wise differentiation in the
        input layer
        :param feature_size: embeddings size of binary features


        '''
        super(CDCorefScorer, self).__init__()
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

        self.char_lstm = nn.LSTM(input_size=char_embedding.shape[1],hidden_size= self.char_hidden_dim,num_layers=1,
                                 bidirectional=False)

        # binary features for coreferring arguments/predicates
        self.coref_role_embeds = nn.Embedding(2,feature_size)

        self.use_mult = use_mult
        self.use_diff = use_diff
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
        char_hidden = self.init_char_hidden(device)
        input_char_seq = self.prepare_chars_seq(seq, device)
        char_embeds = self.char_embeddings(input_char_seq).view(len(seq), 1, -1)
        char_lstm_out, char_hidden = self.char_lstm(char_embeds, char_hidden)
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




