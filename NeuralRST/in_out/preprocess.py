import os
import numpy as np
import torch
import numpy as np

from NeuralRST.models.metric import Metric
from NeuralRST.models.alphabet import Alphabet
from NeuralRST.in_out.util import lower_with_digit_transform
from NeuralRST.transition.state import CState
from torch.autograd import Variable

def construct_embedding_table(alpha, hidden_size, freeze, pretrained_embed = None):
    if alpha is None:
        return None
    scale = np.sqrt(6.0 / (alpha.size()+hidden_size))
    table = np.empty([alpha.size(), hidden_size], dtype=np.float32)
    for word, index, in alpha.alpha2id.items():
        if pretrained_embed is not None:
            if word in pretrained_embed:
                embedding = pretrained_embed[word]
            elif word.lower() in pretrained_embed:
                embedding = pretrained_embed[word.lower()]
            else:
                embedding = np.zeros([1, hidden_size]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, hidden_size]).astype(np.float32)
        else:
            embedding = np.random.uniform(-scale, scale, [1, hidden_size]).astype(np.float32)
        table[index, :] = embedding
    return torch.from_numpy(table)


def create_alphabet(instances, alphabet_directory, logger):
    word_size = 0
    gold_size = 0
        
    word_stat = {}
    tag_stat = {}
    gold_action_stat = {}
    action_label_stat = {}
    etype_stat = {}

    if not os.path.isdir(alphabet_directory):
        print("Creating Alphabets")
        for instance in instances:
            for i in range(len(instance.total_words)):
                word = lower_with_digit_transform(instance.total_words[i].strip())
                tag = instance.total_tags[i]
                word_stat[word] = word_stat.get(word, 0) + 1
                tag_stat[tag] = tag_stat.get(tag, 0) + 1

            for action in instance.gold_actions:
                if (not action.is_shift() and not action.is_finish()):
                    action_label_stat[action.label] = action_label_stat.get(action.label, 0) + 1
                gold_action_stat[action.get_str()] = gold_action_stat.get(action.get_str(), 0) + 1
            
            for k in range(len(instance.edus)):
                etype_stat[instance.edus[k].etype] = etype_stat.get(instance.edus[k].etype, 0) + 1
        
        word_alpha = Alphabet(word_stat, 'word_alpha')
        tag_alpha = Alphabet(tag_stat, 'tag_alpha')
        gold_action_alpha = Alphabet(gold_action_stat, 'gold_action_alpha', for_label_index=True)
        action_label_alpha = Alphabet(action_label_stat, 'action_label_alpha', for_label_index=True)
        etype_alpha = Alphabet(etype_stat, 'etype_alpha')

        word_alpha.save(alphabet_directory)
        tag_alpha.save(alphabet_directory)
        gold_action_alpha.save(alphabet_directory)
        action_label_alpha.save(alphabet_directory)
        etype_alpha.save(alphabet_directory)
    else:
        print("Loading Alphabets")
        word_alpha = Alphabet(word_stat, 'word_alpha')
        tag_alpha = Alphabet(tag_stat, 'tag_alpha')
        gold_action_alpha = Alphabet(gold_action_stat, 'gold_action_alpha')
        action_label_alpha = Alphabet(action_label_stat, 'action_label_alpha')
        etype_alpha = Alphabet(etype_stat, 'etype_alpha')
        
        word_alpha.load(alphabet_directory)
        tag_alpha.load(alphabet_directory)
        gold_action_alpha.load(alphabet_directory, for_label_index=True)
        action_label_alpha.load(alphabet_directory, for_label_index=True)
        etype_alpha.load(alphabet_directory)

    logger.info("Word alphabet size: " + str(word_alpha.size()))
    logger.info("Tag alphabet size: " + str(tag_alpha.size()))
    logger.info("Gold action alphabet size: " + str(gold_action_alpha.size()))
    logger.info("Action Label alphabet size: " + str(action_label_alpha.size()))
    logger.info("Etype alphabet size: " + str(etype_alpha.size()))
    return word_alpha, tag_alpha, gold_action_alpha, action_label_alpha, etype_alpha


def validate_gold_actions(instances, maxStateSize):
    shift_num = 0; reduce_nn_num = 0; reduce_ns_num = 0; reduce_sn_num = 0
    span = Metric(); nuclear = Metric(); relation = Metric(); full = Metric()

    for inst in instances:
        for ac in inst.gold_actions:
            if ac.is_shift():
                shift_num+=1
            if ac.is_reduce():
                if ac.nuclear == 'NN':
                    reduce_nn_num += 1
                elif ac.nuclear == 'NS':
                    reduce_ns_num += 1
                elif ac.nuclear == 'SN':
                    reduce_sn_num += 1
                else:
                    raise Exception('Reduce error, this must have nuclearity')
                # something is here
                assert(ac.label_id != -1)

    print("Reduce NN: " + str(reduce_nn_num))
    print("Reduce NS: " + str(reduce_ns_num))
    print("Reduce SN: " + str(reduce_sn_num))
    print("Shift: " + str(shift_num))

    print("Checking the gold Actions, it will be interrupted if there is error assertion")
    # all_states = [CState() for i in range(maxStateSize)]
    # for inst in instances:
        # step = 0
        # gold_actions = inst.gold_actions
        # action_size = len(gold_actions)
        # all_states[0].ready(inst)
        # while(not all_states[step].is_end()):
            # assert(step < action_size)
            # all_states[step+1] = all_states[step].move(all_states[step+1], gold_actions[step])
            # step += 1
        # assert(step == action_size)
        # result = all_states[step].get_result()
        # span, nuclear, relation, full = inst.evaluate(result, span, nuclear, relation, full)
        # if not span.bIdentical() or not nuclear.bIdentical() or not relation.bIdentical() or not full.bIdentical():
            # raise Exception('Error state conversion!! ')

def get_max_parameter(instances):
    max_edu_size = 0
    max_sent_size = 0
    max_state_size = 0
    
    for instance in instances:
        len_state = len(instance.gold_actions)
        if len_state > max_state_size:
            max_state_size = len_state
        len_edu = len(instance.edus)
        if len_edu > max_edu_size:
            max_edu_size = len_edu
        for edu in instance.edus:
            len_sent = len(edu.words)
            if len_sent > max_sent_size:
                max_sent_size = len_sent
    return max_edu_size, max_sent_size, max_state_size

def batch_data_variable(data, indices, vocab, config, is_training=True):
    batch_size  = len(indices)
    indices = indices.tolist()
    batch = data[indices]
    max_edu_len = -1
    max_edu_num = -1
    for data in batch:
        edu_num = len(data.edus)
        if edu_num > max_edu_num: max_edu_num = edu_num
        for edu in data.edus:
            edu_len = len(edu.words)
            if edu_len > max_edu_len: max_edu_len = edu_len

    edu_words = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_types = Variable(torch.LongTensor(batch_size, max_edu_num).zero_(), requires_grad=False)
    edu_syntax = np.zeros([batch_size, max_edu_num, max_edu_len, config.syntax_dim], dtype=np.float32)
    word_mask = Variable(torch.Tensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_tags = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_mask = Variable(torch.Tensor(batch_size, max_edu_num).zero_(), requires_grad=False)
    word_denominator = Variable(torch.ones(batch_size, max_edu_num).type(torch.FloatTensor) * -1, requires_grad=False)
    gold_action_var = np.ones([batch_size, config.max_state_size], dtype=np.int64)  * (vocab.gold_action_alpha.size())
    len_edus = np.zeros([batch_size], dtype=np.int64)

    for idx in range(batch_size):
        for idy in range(len(batch[idx].edus)):
            len_edus[idx] = len(batch[idx].edus)
            edu = batch[idx].edus[idy]
            edu_mask[idx, idy] = 1
            edu_types[idx, idy] = vocab.etype_alpha.word2id(edu.etype)
            edu_len = len(edu.words)
            word_denominator[idx, idy] = edu_len
            for idz in range(edu_len):
                word = edu.words[idz]
                tag = edu.tags[idz]
                edu_words[idx, idy, idz] = vocab.word_alpha.word2id(word)
                edu_tags[idx, idy, idz] = vocab.tag_alpha.word2id(tag)
                edu_syntax[idx, idy, idz] = edu.syntax_features[idz]
                word_mask[idx, idy, idz] = 1
    
        if is_training:
            max_gold = len(batch[idx].gold_actions)
            if max_gold > config.max_state_size: max_gold = config.max_state_size
            for idy in range(max_gold):
                gold_action_str = batch[idx].gold_actions[idy].get_str()
                gold_action_var[idx][idy] = vocab.gold_action_alpha.word2id(gold_action_str)
    gold_action_var = Variable(torch.from_numpy(gold_action_var), volatile=False, requires_grad=False)
    edu_syntax = Variable(torch.from_numpy(edu_syntax), volatile=False, requires_grad=False)
    if config.use_gpu:
        edu_words = edu_words.cuda()
        edu_tags = edu_tags.cuda()
        edu_types = edu_types.cuda()
        edu_mask = edu_mask.cuda()
        word_mask = word_mask.cuda()
        gold_action_var = gold_action_var.cuda()
        word_denominator = word_denominator.cuda()
        edu_syntax = edu_syntax.cuda()
    
    return edu_words, edu_tags, edu_types, edu_mask, word_mask, gold_action_var, len_edus, word_denominator, edu_syntax



    
    
