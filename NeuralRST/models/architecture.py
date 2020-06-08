__author__ = 'fajri'

import copy
import operator
import numpy as np
from collections import deque as queue
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.rnn as rnn
from torch.autograd import Variable
from NeuralRST.in_out.instance import CResult
from NeuralRST.models.metric import Metric
from NeuralRST.models.explorer import Explorer
from NeuralRST.modules.embedding import Embedding
from NeuralRST.modules.variational_rnn import VarMaskedLSTM, VarMaskedFastLSTM
from NeuralRST.transition.state import CState
from NeuralRST.transition.action import CAction
from NeuralRST.modules.layer import *

class MainArchitecture(nn.Module):
    def __init__(self, vocab, config, embedd_word=None, embedd_tag=None, embedd_etype=None):
        super(MainArchitecture, self).__init__()
        random.seed(999)
        
        num_embedding_word = vocab.word_alpha.size()
        num_embedding_tag = vocab.tag_alpha.size()
        num_embedding_etype = vocab.etype_alpha.size()
        self.word_embedd = Embedding(num_embedding_word, config.word_dim, embedd_word)
        self.tag_embedd = Embedding(num_embedding_tag, config.tag_dim, embedd_tag)
        self.etype_embedd = Embedding(num_embedding_etype, config.etype_dim, embedd_etype)
        
        self.config = config
        self.vocab = vocab

        dim_enc1 = config.word_dim + config.tag_dim
        dim_enc2 = config.syntax_dim
        dim_enc3 = config.hidden_size * 4 + config.etype_dim

        self.rnn_word_tag = MyLSTM(input_size=dim_enc1, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout_in=config.drop_prob, dropout_out=config.drop_prob)
        self.rnn_syntax = MyLSTM(input_size=dim_enc2, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout_in=config.drop_prob, dropout_out=config.drop_prob)
        self.rnn_edu = MyLSTM(input_size=dim_enc3, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True, bidirectional=True, dropout_in=config.drop_prob, dropout_out=config.drop_prob)

        self.dropout_in = nn.Dropout(p=config.drop_prob)
        self.dropout_out = nn.Dropout(p=config.drop_prob)
        out_dim1 = config.hidden_size * 2
        out_dim2 = config.hidden_size * 2
        
        dim_dec = vocab.gold_action_alpha.size()
        self.mlp_layer = NonLinear(out_dim2 * 4, config.hidden_size, activation=nn.LeakyReLU(0.1))
        self.output = nn.Linear(config.hidden_size, dim_dec)

        self.metric = Metric()
        self.explorer = Explorer(vocab)

        self.batch_states = []
        self.batch_steps = []
        for idx in range(config.batch_size):
            self.batch_states.append([])
            self.batch_steps.append(0)
            for idy in range(1024):
                self.batch_states[idx].append(CState())

        self.training = True
    
    def run_rnn_word_tag(self, input_word, input_tag, word_mask):
        word = self.word_embedd(input_word)
        tag = self.tag_embedd(input_tag)
        word = self.dropout_in(word)
        tag = self.dropout_in(tag)
        
        # apply rnn over EDU
        tensor = torch.cat([word, tag], dim=-1)
        batch_size, edu_size, word_in_edu, hidden_size = tensor.shape
        tensor = tensor.view(batch_size * edu_size, word_in_edu, hidden_size)
        word_mask = word_mask.view(batch_size * edu_size, word_in_edu)
        tensor, hn = self.rnn_word_tag(tensor, word_mask, None)
        
        tensor = tensor.transpose(0,1).contiguous()
        tensor = tensor.view(batch_size, edu_size, word_in_edu, -1)
        # tensor = self.dropout_out(tensor)
        return tensor

    def run_rnn_syntax(self, syntax, word_mask):
        syntax = self.dropout_in(syntax)
        
        # apply rnn over EDU
        batch_size, edu_size, word_in_edu, hidden_size = syntax.shape
        syntax = syntax.view(batch_size * edu_size, word_in_edu, hidden_size)
        word_mask = word_mask.view(batch_size * edu_size, word_in_edu)
        tensor, hn = self.rnn_syntax(syntax, word_mask, None)
        
        tensor = tensor.transpose(0,1).contiguous()
        tensor = tensor.view(batch_size, edu_size, word_in_edu, -1)
        # tensor = self.dropout_out(tensor)
        return tensor
    
    def run_rnn_edu(self, word_representation, syntax_representation, word_denominator, input_etype, edu_mask):
        etype = self.etype_embedd(input_etype)
        etype = self.dropout_in(etype)
       
        # apply average pooling based on EDU span
        batch_size, edu_size, word_in_edu, hidden_size = word_representation.shape
        word_representation = word_representation.view(batch_size * edu_size, word_in_edu, -1)
        edu_representation1 = AvgPooling(word_representation, word_denominator.view(-1))
        edu_representation1 = edu_representation1.view(batch_size, edu_size, -1)
        
        syntax_representation = syntax_representation.view(batch_size * edu_size, word_in_edu, -1)
        edu_representation2 = AvgPooling(syntax_representation, word_denominator.view(-1))
        edu_representation2 = edu_representation2.view(batch_size, edu_size, -1)

        edu_representation = torch.cat([edu_representation1, edu_representation2, etype], dim=-1)
        output, hn = self.rnn_edu(edu_representation, edu_mask, None)
        output = output.transpose(0,1).contiguous()
        # output = self.dropout_out(output)
        return output

    def forward_all(self, input_word, input_tag, input_etype, edu_mask, word_mask, word_denominator, syntax):
        word_tag_output = self.run_rnn_word_tag(input_word, input_tag, word_mask)
        syntax_output = self.run_rnn_syntax(syntax, word_mask)
        tensor = self.run_rnn_edu(word_tag_output, syntax_output, word_denominator, input_etype, edu_mask) 
        return tensor

    def flatten_layer(self, hidden_state, cut=None):
        mlp_hidden = self.mlp_layer(hidden_state)
        # mlp_hidden = self.activation(self.mlp_layer(hidden_state))
        action_score = self.output(mlp_hidden)
        if cut is not None:
            action_score += cut
        return action_score

    def not_all_finished(self, batch_size):
        for idx in range(batch_size):
            cur_step = self.batch_steps[idx]
            if not self.batch_states[idx][cur_step].is_end():
                return True
        return False

    def get_candidate_and_feats(self, batch_size):
        candidates = []
        feats = []
        for idx in range(batch_size):
            step = self.batch_steps[idx]
            state = self.batch_states[idx][step]
            if state.is_end():
                candidates.append([None])
                feats.append([None])
            else:
                candidates.append([state.get_candidate_actions(self.vocab)])
                feats.append([state.prepare_index()])
        return candidates, feats
    
    def prepare_hidden_size(self, feats, encoder_output):
        batch_size, edu_num, hidden = encoder_output.shape
        bucket = Variable(torch.zeros(batch_size, 1, hidden)).type(torch.FloatTensor)
        if self.config.use_gpu:
            bucket = bucket.cuda()
        edu_rep = torch.cat((encoder_output, bucket), 1) # batch_size, action_num + 1, hidden_size
        edu_rep = edu_rep.view(batch_size * (edu_num + 1), hidden)

        stack_index = Variable(torch.ones(batch_size * 3 * edu_num)).type(torch.LongTensor) * edu_num
        stack_denominator = Variable(torch.ones(batch_size * 3)).type(torch.FloatTensor) * -1
        queue_index = Variable(torch.ones(batch_size)).type(torch.LongTensor) * edu_num

        for idx in range(batch_size):
            feat = feats[idx][0]
            feat_offset = idx * (edu_num + 1)
            stack_offset = idx * 3 * edu_num
            denominator_offset = idx * 3
            if feat is None:
                continue
            if feat.q0 is not None:
                queue_index[idx] = feat_offset + feat.q0
            if feat.s0 is not None:
                edu_start = feat.s0.edu_start
                edu_end = feat.s0.edu_end
                l = edu_end - edu_start + 1
                index_offset = stack_offset + (0 * edu_num)
                stack_denominator [denominator_offset + 0] = l
                for j in range(l):
                    stack_index[index_offset + j] = feat_offset + j + edu_start
            if feat.s1 is not None:
                edu_start = feat.s1.edu_start
                edu_end = feat.s1.edu_end
                l = edu_end - edu_start + 1
                index_offset = stack_offset + (1 * edu_num)
                stack_denominator [denominator_offset + 1] = l
                for j in range(l):
                    stack_index[index_offset + j] = feat_offset + j + edu_start
            if feat.s2 is not None:
                edu_start = feat.s2.edu_start
                edu_end = feat.s2.edu_end
                l = edu_end - edu_start + 1
                index_offset = stack_offset + (2 * edu_num)
                stack_denominator [denominator_offset + 2] = l
                for j in range(l):
                    stack_index[index_offset + j] = feat_offset + j + edu_start
        
        if self.config.use_gpu:
            queue_index = queue_index.cuda()
            stack_index = stack_index.cuda()
            stack_denominator = stack_denominator.cuda()
        
        queue_state = torch.index_select(edu_rep, 0, queue_index)
        stack_state = torch.index_select(edu_rep, 0, stack_index)

        stack_state = stack_state.view(batch_size * 3, edu_num, hidden)
        stack_state = AvgPooling(stack_state, stack_denominator)
        #hidden_state = F.max_pool1d(edu_state.transpose(2, 1), kernel_size=EDU_num).squeeze(-1)

        queue_state = queue_state.view(batch_size, 1, hidden)
        stack_state = stack_state.view(batch_size, 3, hidden)
        hidden_state = torch.cat([stack_state, queue_state], -2)
        hidden_state = hidden_state.view(batch_size, -1)

        return hidden_state

    def get_cut(self, batch_size, candidates, feats):
        action_size = self.vocab.gold_action_alpha.size()
        cut_data = np.array([[0] * action_size] * batch_size, dtype=float)
        for idx in range(batch_size):
            step = self.batch_steps[idx]
            state = self.batch_states[idx][step]
            if candidates[idx][0] is not None:
                cut_data[idx] = candidates[idx][0] * -1e20
        cut = Variable(torch.from_numpy(cut_data).type(torch.FloatTensor))
        if self.config.use_gpu:
            cut = cut.cuda()
        return cut

    def get_prediction_and_gold(self, batch_size, scores, gold_actions):
        assert batch_size == len(scores)
        predicted_actions = []
        golds = []

        for idx in range(batch_size):
            state = self.batch_states[idx]
            step = self.batch_steps[idx]
            pred_id = np.argmax(scores[idx])
            pred_action = self.vocab.id2action[pred_id]
            predicted_actions.append(pred_action)
            
            if self.training:
                if not state[step].is_end():
                    gold_id = gold_actions[idx][step].data[0]
                    gold = self.vocab.id2action[gold_id]
                    golds.append(gold)
                    self.metric.overall_label_count += 1
                    if pred_id == gold_id:
                        self.metric.correct_label_count += 1
                else:
                    golds.append(None)
        return predicted_actions, golds

    def move(self, actions):
        batch_size = len(actions)
        for idx in range(batch_size):
            state = self.batch_states[idx]
            step = self.batch_steps[idx]
            if not state[step].is_end():
                next_state = self.batch_states[idx][step + 1]
                state[step].move(next_state, actions[idx])
                self.batch_steps[idx] += 1
    
    def compute_loss(self, decoder_output, gold_actions):
        batch_size, action_len, action_num = decoder_output.shape
        idx_ignore = self.vocab.gold_action_alpha.size()
        loss = F.cross_entropy(decoder_output.view(batch_size * action_len, action_num),
                        gold_actions[:,:action_len].contiguous().view(batch_size * action_len),
                        ignore_index = idx_ignore)
        return loss
    
    def get_result(self, batch_size):
        results = []
        for idx in range(batch_size):
            step = self.batch_steps[idx]
            state = self.batch_states[idx][step]
            results.append(state.get_result())
        return results

    # STATIC and DYNAMIC ORACLE
    def decode(self, encoder_output, gold_action_ids, gold_subtrees, len_edus):
        batch_size, edu_size, hidden_size = encoder_output.shape
        
        for idx in range(batch_size):
            self.batch_steps[idx] = 0
            self.batch_states[idx][0].clear()
            self.batch_states[idx][0].ready(len_edus[idx]) #set ready
            #Increase batch state stepsize from 1024
            if len_edus*2>1024:
                for idy in range(len_edus*2-1024+1):
                    self.batch_states[idx].append(CState())
 
        all_decoder_output = []
        optimal_action_ids = []
        while(self.not_all_finished(batch_size)):
            candidates, feats = self.get_candidate_and_feats(batch_size)
            cut = self.get_cut(batch_size, candidates, feats)
            hidden_state = self.prepare_hidden_size(feats, encoder_output)
            decoder_output = self.flatten_layer(hidden_state, cut)
            decoder_score = decoder_output.data.cpu().numpy()
            predicted_actions, golds = self.get_prediction_and_gold(batch_size, decoder_score, gold_action_ids)
            if self.training:
                if self.config.flag_oracle:
                    optimal_actions, optimal_action_id = self.get_oracle_actions(gold_subtrees)
                    optimal_action_ids.append(optimal_action_id)
                    p = random.random()
                    if self.config.oracle_prob > p:
                        self.move(optimal_actions)
                    else:
                        self.move(predicted_actions)
                else:
                    self.move(golds)
            else:
                self.move(predicted_actions)
            all_decoder_output.append(decoder_output.view(batch_size, 1, -1))

        # Evaluation
        all_decoder_output = torch.cat(all_decoder_output, dim=1)
        if self.training and not self.config.flag_oracle:
            return self.compute_loss(all_decoder_output, gold_action_ids)
        elif self.training and self.config.flag_oracle:
            optimal_action_ids = torch.cat(optimal_action_ids, dim=1)
            return self.compute_loss(all_decoder_output, optimal_action_ids)
        else:
            return self.get_result(batch_size)

    def loss(self, subset_data, gold_subtrees):
        words, tags, etypes, edu_mask, word_mask, gold_actions, len_edus, word_denominator, syntax = subset_data
        encoder_output = self.forward_all(words, tags, etypes, edu_mask, word_mask, word_denominator, syntax)
        if self.training:
            cost = self.decode(encoder_output, gold_actions, gold_subtrees, len_edus)
            return cost, cost.data[0]
        else:
            results = self.decode(encoder_output, gold_actions, gold_subtrees, len_edus)
            return results
    
    def get_oracle_actions(self, gold_subtrees):
        batch_size = len(self.batch_states)
        optimal_actions = []
        optimal_action_ids = []
        for b_iter in range(batch_size):
            step = self.batch_steps[b_iter]
            state = self.batch_states[b_iter][step]
            if not state.is_end():
                optimal_action = self.explorer.get_oracle(state, gold_subtrees[b_iter].subtrees)
                optimal_action_id = [self.vocab.gold_action_alpha.alpha2id[optimal_action.get_str()]]
                optimal_actions.append(optimal_action)
                optimal_action_ids.append(optimal_action_id)
            else:
                optimal_actions.append(None)
                optimal_action_ids.append([self.vocab.gold_action_alpha.size()])
        optimal_action_ids = np.array(optimal_action_ids, dtype=np.int64)
        optimal_action_ids = Variable(torch.from_numpy(optimal_action_ids), volatile=False, requires_grad=False)
        if self.config.use_gpu:
            optimal_action_ids = optimal_action_ids.cuda()
        return optimal_actions, optimal_action_ids

