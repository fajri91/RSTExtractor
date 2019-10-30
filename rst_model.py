import numpy as np
import torch
from torch.autograd import Variable

from NeuralRST.in_out.util import load_embedding_dict, get_logger
from NeuralRST.in_out.preprocess import create_alphabet
from NeuralRST.in_out.preprocess import batch_data_variable
from NeuralRST.models.vocab import Vocab
from NeuralRST.models.metric import Metric
from NeuralRST.models.config import Config
from NeuralRST.models.architecture import MainArchitecture


class RSTModel(object):
    def __init__(self, rst_config_path):
        print("................................................")
        print("LOADING RST Model")
        self.config = Config(None)
        self.config.load_config(rst_config_path)
        self.logger = get_logger("RSTParser RUN", self.config.use_dynamic_oracle, self.config.model_path)
        word_alpha, tag_alpha, gold_action_alpha, action_label_alpha, etype_alpha = create_alphabet(None, self.config.alphabet_path, self.logger)
        self.vocab = Vocab(word_alpha, tag_alpha, etype_alpha, gold_action_alpha, action_label_alpha)
        self.network = MainArchitecture(self.vocab, self.config) 
        self.network.load_state_dict(torch.load(self.config.model_name))
        if self.config.use_gpu:
            self.network = self.network.cuda()
        self.network.eval()

    def prepare_data(self, batch, batch_size):
        config = self.config
        vocab = self.vocab
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
        edu_syntax = Variable(torch.Tensor(batch_size, max_edu_num, max_edu_len, config.syntax_dim).zero_(), requires_grad=False)
        word_mask = Variable(torch.Tensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
        edu_tags = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
        edu_mask = Variable(torch.Tensor(batch_size, max_edu_num).zero_(), requires_grad=False)
        word_denominator = Variable(torch.ones(batch_size, max_edu_num).type(torch.FloatTensor) * -1, requires_grad=False)
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
                    edu_syntax[idx, idy, idz] = edu.syntax_features[idz].view(config.syntax_dim)
                    word_mask[idx, idy, idz] = 1
        
        if config.use_gpu:
            edu_words = edu_words.cuda()
            edu_tags = edu_tags.cuda()
            edu_types = edu_types.cuda()
            edu_mask = edu_mask.cuda()
            word_mask = word_mask.cuda()
            word_denominator = word_denominator.cuda()
            edu_syntax = edu_syntax.cuda()
        
        return edu_words, edu_tags, edu_types, edu_mask, word_mask, len_edus, word_denominator, edu_syntax

    def get_edu_representation(self, data_test):
        words, tags, etypes, edu_mask, word_mask, len_edus, word_denominator, syntax = data_test
        encoder_output = self.network.forward_all(words, tags, etypes, edu_mask, word_mask, word_denominator, syntax)
        return encoder_output

    def get_subtree(self, data_test):
        words, tags, etypes, edu_mask, word_mask, len_edus, word_denominator, syntax = data_test
        self.network.training = False
        encoder_output = self.network.forward_all(words, tags, etypes, edu_mask, word_mask, word_denominator, syntax)
        results = self.network.decode(encoder_output, [], [], len_edus)
        return results



