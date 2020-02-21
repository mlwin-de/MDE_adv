#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.mde_vector_number = 8

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'MDE':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.entity_embedding2 = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.entity_embedding3 = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.entity_embedding4 = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.entity_embedding5 = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.entity_embedding6 = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.entity_embedding7 = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.entity_embedding8 = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding2,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding3,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding4,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding5,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding6,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding7,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.entity_embedding8,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.relation_embedding2 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.relation_embedding3 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.relation_embedding4 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.relation_embedding5 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.relation_embedding6 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.relation_embedding7 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.relation_embedding8 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding2,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding3,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding4,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding5,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding6,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding7,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.relation_embedding8,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'MDE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single' and self.model_name != 'MDE':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        elif mode == 'single' and self.model_name == 'MDE':
            batch_size, negative_sample_size = sample.size(0), 1

            h = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h2 = torch.index_select(
                self.entity_embedding2,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t2 = torch.index_select(
                self.entity_embedding2,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r5 = torch.index_select(
                self.relation_embedding5,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h6 = torch.index_select(
                self.entity_embedding6,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r6 = torch.index_select(
                self.relation_embedding6,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t6 = torch.index_select(
                self.entity_embedding6,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r7 = torch.index_select(
                self.relation_embedding7,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            h8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            r8 = torch.index_select(
                self.relation_embedding8,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            t8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            head = [h, h2, h3, h4, h5,h6,h7,h8]
            relation = [r, r2, r3, r4,r5,r6,r7,r8]
            tail = [t, t2, t3, t4, t5,t6,t7,t8]

        elif mode == 'head-batch' and self.model_name != 'MDE':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch' and self.model_name != 'MDE':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'head-batch' and self.model_name == 'MDE':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            h = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h2 = torch.index_select(
                self.entity_embedding2,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t2 = torch.index_select(
                self.entity_embedding2,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r5 = torch.index_select(
                self.relation_embedding5,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h6 = torch.index_select(
                self.entity_embedding6,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r6 = torch.index_select(
                self.relation_embedding6,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t6 = torch.index_select(
                self.entity_embedding6,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r7 = torch.index_select(
                self.relation_embedding7,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            h8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            r8 = torch.index_select(
                self.relation_embedding8,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            t8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            head = [h, h2, h3, h4, h5,h6,h7,h8]
            relation = [r, r2, r3, r4,r5,r6,r7,r8]
            tail = [t, t2, t3, t4, t5,t6,t7,t8]

        elif mode == 'tail-batch' and self.model_name == 'MDE':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            h = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h2 = torch.index_select(
                self.entity_embedding2,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r2 = torch.index_select(
                self.relation_embedding2,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t2 = torch.index_select(
                self.entity_embedding2,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r3 = torch.index_select(
                self.relation_embedding3,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t3 = torch.index_select(
                self.entity_embedding3,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r4 = torch.index_select(
                self.relation_embedding4,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t4 = torch.index_select(
                self.entity_embedding4,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            h5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r5 = torch.index_select(
                self.relation_embedding5,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t5 = torch.index_select(
                self.entity_embedding5,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h6 = torch.index_select(
                self.entity_embedding6,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r6 = torch.index_select(
                self.relation_embedding6,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t6 = torch.index_select(
                self.entity_embedding6,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r7 = torch.index_select(
                self.relation_embedding7,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t7 = torch.index_select(
                self.entity_embedding7,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            h8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            r8 = torch.index_select(
                self.relation_embedding8,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            t8 = torch.index_select(
                self.entity_embedding8,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            head = [h, h2, h3, h4, h5,h6,h7,h8]
            relation = [r, r2, r3, r4,r5,r6,r7,r8]
            tail = [t, t2, t3, t4, t5,t6,t7,t8]
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'MDE': self.MDE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def MDE(self, h, r, t, mode):
        #psi = self.gamma.item()#1.2
        # a = h + r - t
        # b = h + t - r
        # c = t + r - h
        # d = h - r * t
        if mode == 'head-batch':
            a = h[0] + (r[0] - t[0])
            b = h[1] + (t[1] - r[1])
            c = t[2] + (r[2] - h[2])
            d = h[3] - (r[3] * t[3])

            e = h[4] + (r[4] - t[4])
            f = h[5] + (t[5] - r[5])
            g = t[6] + (r[6] - h[6])
            i = h[7] - (r[7] * t[7])
        else:
            a = (h[0] + r[0]) - t[0]
            b = (h[1] + t[1]) - r[1]
            c = (t[2] + r[2]) - h[2]
            d = h[3] - (r[3] * t[3])#(h[3] - t[3]) * r[3]#t[3] - (r[3] * h[3])  #t[3]- (h[3] * r[3])

            e = (h[4] + r[4]) - t[4]
            f = (h[5] + t[5]) - r[5]
            g = (t[6] + r[6]) - h[6]
            i = h[7] - (r[7] * t[7])#(h[7] - t[7]) *r[7] #t[7] - (r[7] * h[7])# - t[7]


        # print(mode)
        score_a = (torch.norm(a, p=2, dim=2) + torch.norm((e), p=2, dim=2)) / 2.0
        score_b = (torch.norm((b), p=2, dim=2)  + torch.norm((f), p=2, dim=2)) / 2.0
        score_c = (torch.norm((c), p=2, dim=2)   + torch.norm((g), p=2, dim=2)) / 2.0
        score_d = (torch.norm((d), p=2, dim=2)  + torch.norm((i), p=2, dim=2)) / 2.0
        score_all = (1.5 * score_a + 3.0 * score_b + 1.5 * score_c + 3.0 * score_d) / 9.0 #- psi
        #score_all = score_a - .5 * score_b + score_c  #- psi
        #score_all =  .1077 * score_a - .0553 * score_b + 0.0064 *  score_c - .145 * score_d + 0.001

        #score_all = score_a - .5 * score_b + score_c - .2 * score_d
        score = self.gamma.item() - score_all
        # (.6 * score_a + .4* score_b + .6 * score_c) * 2#) - (3.0/9.0)*score_d
        # score = self.gamma.item() - score  # torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

            y = Variable((torch.Tensor([-1]))).cuda()
            lambda_pos = Variable(torch.FloatTensor([args.gamma_1])).cuda()
            lambda_neg = Variable(torch.FloatTensor([args.gamma_2])).cuda()
        else:
            y = Variable((torch.Tensor([-1])))
            lambda_pos = Variable(torch.Tensor([args.gamma_1]))
            lambda_neg = Variable(torch.Tensor([args.gamma_2]))
        beta_1 = args.beta_1
        beta_2 = args.beta_2

        if args.mde_score:
            negative_score = - model((positive_sample, negative_sample), mode=mode)

            if args.negative_adversarial_sampling:
                negative_score = (negative_score.sum(dim=1) - (args.negative_sample_size * negative_sample.shape[
                    0])) * args.adversarial_temperature  # - (args.negative_sample_size * args.gamma_2)
            else:
                negative_score = negative_score.mean(dim=1)

            positive_score = - model(positive_sample)
            if args.uni_weight:
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            positive_sample_loss = positive_sample_loss.unsqueeze(dim=0)
            negative_sample_loss = negative_sample_loss.unsqueeze(dim=0)
            loss, positive_sample_loss, negative_sample_loss = model.mde_loss_func(positive_sample_loss,
                                                                                   negative_sample_loss, y, lambda_pos,
                                                                                   lambda_neg, beta_1, beta_2)
            # loss = (positive_sample_loss + negative_sample_loss) / 2

            if args.regularization != 0.0:
                # Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                        model.entity_embedding.norm(p=3) ** 3 +
                        model.relation_embedding.norm(p=3).norm(p=3) ** 3
                )
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}

            loss.backward()

            optimizer.step()

            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }

            return log

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def mde_loss_func(p_score, n_score, y, lambda_pos, lambda_neg, beta_1, beta_2):
        criterion = nn.MarginRankingLoss(1.0, False)

        pos_loss = criterion(p_score, lambda_pos, y)
        neg_loss = criterion(n_score, lambda_neg, -y)
        loss = beta_1 * pos_loss + beta_2 * neg_loss
        return loss, pos_loss, neg_loss

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                if args.model == "MDE":
                    y_score = model(sample).squeeze(1).cpu().numpy()
                else:
                    y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)
                        if args.model == "MDE":
                            score = model((positive_sample, negative_sample), mode)
                        else:
                            score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
