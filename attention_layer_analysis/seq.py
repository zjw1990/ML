# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs






def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))

    return result


def variablesFromPair(pair):

    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru= nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #print(input)
        embedded = self.embedding(input).view(1, 1, -1)

# output is 1*1*256
        output = embedded
# hidden is 1*1*256
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,max_length = MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, 1, -1)  # calculate input vectors is [SOS,target_sentence]

# embedded 1*256
        embedded = self.dropout(embedded)   # dropout
# atten_eti          1*10
        attention_eti = self.attn(torch.cat((embedded[0], hidden[0]), 1))  # a neural network feed by s(t-1) and h(i)
        # print("eti is           ",attention_eti)
# attn_weight         1*10
        attn_weights = F.softmax(attention_eti, dim=1)  # attention weight alpha where alpha is a softmax of eti
        # print("alpah is         ",attn_weights)
        # print(encoder_outputs.unsqueeze(0))
        # print(attn_weights.unsqueeze(0))
# attn_weights.s   1*1*10   encoder_output.s  1*10*256   attn_applied/c  1*1*256
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))  # c(t) which is the context vector of time t in decoder
# output: 1*512
        output = torch.cat((embedded[0], attn_applied[0]), 1) #
# output: 1*256
        output = self.attn_combine(output).unsqueeze(0) # a layer feed by x and context vector
# output: 1*256
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
# output : 1*lang2_nwords
        output = F.log_softmax(self.out(output[0]), dim=1) # a layer feed by output
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result






encoder = EncoderRNN(input_lang.n_words,256)
pair = random.choice(pairs)
encoder_inputs,target = variablesFromPair(pair)
encoder_input_length = encoder_inputs.size()[0]
target_length = target.size()[0]
# 1*1*256
encoder_hidden = encoder.initHidden()
# lengh*256
encoder_outputs = Variable(torch.zeros(encoder_input_length,encoder.hidden_size))
for ei in range(encoder_input_length):
    # print(encoder_hidden)
    encoder_output, encoder_hidden = encoder(encoder_inputs[ei],encoder_hidden)
    # print(encoder_output)
# encoder_output 1*1*256
    # print("encoder_output is       ", encoder_output)
    # print("encoder_outputs is       ", encoder_outputs)
    encoder_outputs[ei] = encoder_output[0][0]
# print(encoder_outputs)
# print(encoder_hidden)
# 1*1
decoder_input = Variable(torch.LongTensor([[SOS_token]]))
# 1*1*256
decoder_hidden = encoder_hidden

teacher_forcing_ratio = 0.5

use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
attention_decoder = AttnDecoderRNN(256,output_size=output_lang.n_words,dropout_p=0.1,max_length=encoder_input_length)
if use_teacher_forcing:
    for di in range(target_length):
        #print("tf   decoder_input is  ",decoder_input)
        # print(output_lang.index2word[decoder_input.data])

        decoder_output, decoder_hidden, decoder_attention = attention_decoder.forward(decoder_input,decoder_hidden,encoder_outputs)

        decoder_input = target[di]
else:
    for di in range(target_length):
# decoder_output: lang2_nwords
        #print("n_tf decoder_input is   ",decoder_input)
        # print(output_lang.index2word[decoder_input.data])
        decoder_output, decoder_hidden, decoder_attention = attention_decoder.forward(decoder_input,decoder_hidden,encoder_outputs)
        topval, topidx = decoder_output.data.topk(1)
        # print("top i is    ",topidx,"topv is          ",topval)
        ni = topidx[0][0]
        # print("ni is   ",ni)
        decoder_input = Variable(torch.LongTensor([[ni]]))

























