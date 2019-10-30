import os
import torch
import json
import numpy as np
from torch.autograd import Variable
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.io import CoNLLXWriter, utils
from neuronlp2.tasks import parser
from neuronlp2.models import BiRecurrentConvBiAffine

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
PAD_POS = b"_PAD_POS"
PAD_TYPE = b"_<PAD>"
PAD_CHAR = b"_PAD_CHAR"
ROOT = b"_ROOT"
ROOT_POS = b"_ROOT_POS"
ROOT_TYPE = b"_<ROOT>"
ROOT_CHAR = b"_ROOT_CHAR"
END = b"_END"
END_POS = b"_END_POS"
END_TYPE = b"_<END>"
END_CHAR = b"_END_CHAR"
_START_VOCAB = [PAD, ROOT, END]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140, 200, 300]


class BiaffineModel(object):
    def __init__(self, model_path, model_name):
        print("................................................")
        print("LOADING Biaffine Model")
        alphabet_path = os.path.join(model_path, 'alphabets/')
        model_name = os.path.join(model_path, model_name)

        self.word_alpha, self.char_alpha, self.tag_alpha, self.type_alpha = conllx_data.create_alphabets(alphabet_path, None, data_paths=[None, None], max_vocabulary_size=50000, embedd_dict=None)
        self.id2word = {v: k for k, v in self.word_alpha.instance2index.iteritems()}
        
        num_words = self.word_alpha.size()
        num_chars = self.char_alpha.size()
        num_pos = self.tag_alpha.size()
        num_types = self.type_alpha.size()

        print("Word Alphabet Size: %d" % num_words)
        print("Character Alphabet Size: %d" % num_chars)
        print("POS Alphabet Size: %d" % num_pos)
        print("Type Alphabet Size: %d" % num_types)


        def load_model_arguments_from_json():
            arguments = json.load(open(arg_path, 'r'))
            return arguments['args'], arguments['kwargs']

        arg_path = model_name + '.arg.json'
        args, kwargs = load_model_arguments_from_json()
        self.network = BiRecurrentConvBiAffine(*args, **kwargs)
        self.network.load_state_dict(torch.load(model_name))
        
        self.network.id2word = self.id2word
        self.network.cuda()
        self.network.eval()

    def prepare_data(self, sentences, use_gpu=True):
        ret_value = []
        for sentence in sentences:
            inst_size = sentence.length()
            data = None
            max_len = 0
            bucket = 0
            for bucket_size in _buckets:
                if inst_size < bucket_size:
                    bucket = bucket_size
                    data = [sentence.word_ids, sentence.seq_char_ids, sentence.tag_ids]
                    max_len = max([len(seq_char) for seq_char in sentence.seq_chars])
                    break
            if data is None: # meaning the sentence is too long, we cut it into 300 length
                bucket = _buckets[-1]
                data = [sentence.word_ids[:bucket], sentence.seq_char_ids[:bucket], sentence.tag_ids[:bucket]]
                max_len = max([len(seq_char) for seq_char in sentence.seq_chars])
                

            char_length = min(utils.MAX_CHAR_LENGTH, max_len + utils.NUM_CHAR_PAD)
            wid_inputs = np.empty([1, bucket], dtype=np.int64)
            cid_inputs = np.empty([1, bucket, char_length], dtype=np.int64)
            pid_inputs = np.empty([1, bucket], dtype=np.int64)

            masks = np.zeros([1, bucket], dtype=np.float32)
            single = np.zeros([1, bucket], dtype=np.int64)

            lengths = np.empty(bucket, dtype=np.int64)

            wids = data[0]
            cid_seqs = data[1]
            pids = data[2]
            inst_size = len(wids)
            lengths[0] = inst_size
            # word ids
            wid_inputs[0, :inst_size] = wids
            wid_inputs[0, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                limit = len(cids)
                if limit > char_length: limit = char_length
                try:
                    cid_inputs[0, c, :limit] = cids[:limit]
                    cid_inputs[0, c, limit:] = PAD_ID_CHAR
                except:
                    import ipdb; ipdb.set_trace()
            cid_inputs[0, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[0, :inst_size] = pids
            pid_inputs[0, inst_size:] = PAD_ID_TAG
            # masks
            masks[0, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if self.word_alpha.is_singleton(wid):
                    single[0, j] = 1

            words = Variable(torch.from_numpy(wid_inputs), volatile=False)
            chars = Variable(torch.from_numpy(cid_inputs), volatile=False)
            pos = Variable(torch.from_numpy(pid_inputs), volatile=False)
            masks = Variable(torch.from_numpy(masks), volatile=False)
            single = Variable(torch.from_numpy(single), volatile=False)
            lengths = torch.from_numpy(lengths)
            if use_gpu:
                words = words.cuda()
                chars = chars.cuda()
                pos = pos.cuda()
                masks = masks.cuda()
                single = single.cuda()
                lengths = lengths.cuda()
            index = slice(0,1)
            ret_value.append((words[index], chars[index], pos[index], masks[index], lengths[index], sentence.words, sentence.edu_ids))
        return ret_value

    def get_syntax_feature(self, data_test, sentences):
        sent = 0
        syntax_features = []
        for data in data_test:
            cur_length = len(sentences[sent].words)
            word, char, pos, masks, lengths, original_words, edu_ids = data
            sent += 1
            syntax_feature = self.network.get_syntax_feature(original_words, word, char, pos, mask=masks, length=lengths)
            _ , sent_len, dim = syntax_feature.shape
            if sent_len != cur_length:
                assert sent_len < cur_length
                diff = cur_length - sent_len
                zeros = Variable(torch.zeros(1, diff, dim)).type(torch.FloatTensor).cuda()
                syntax_feature = torch.cat([syntax_feature, zeros], dim=1)
            syntax_features.append(syntax_feature)
        return syntax_features
