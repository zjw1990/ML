# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])



class CBOW(nn.Module):

    def __init__(self,vocab_size,embedding_dim,context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.linear1 = nn.Linear(2*context_size*embedding_dim,128)
        self.linear2 = nn.Linear(128,vocab_size)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        print('embed is', embed)
        embes = embed.view(1,-1)
        print('view is'  ,embes)
        out = F.relu(self.linear1(embes))
        out = self.linear2(out)
        log_prob = F.log_softmax(out,dim=1)
        return log_prob
# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


losses = []
loss_func = nn.NLLLoss()
model = CBOW(vocab_size=vocab_size,embedding_dim=EMBEDDING_DIM,context_size=CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epch in range(50):

    total_loss = torch.Tensor([0])

    for context,target in data:

        context_idx = make_context_vector(context,word_to_ix)
        # form: [13,7,18,42]
        model.zero_grad()

        log_prob = model.forward(context_idx)
        loss = loss_func(log_prob,Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data

    losses.append(total_loss)
print(losses)



