import sys, time
import numpy as np
import random
import re, os
from datetime import datetime
import argparse
from sentence import Sentence, Instance, EDU
from rst_model import RSTModel
from biaffine_model import BiaffineModel
import glob
import threading
import math
from multiprocessing import Process

sys.path.append(".")
ROOT = b"_ROOT"
ROOT_POS = b"_ROOT_POS"
ROOT_CHAR = b"_ROOT_CHAR"
END = b"_END"
END_POS = b"_END_POS"
END_CHAR = b"_END_CHAR"

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(br"\d")
BIAFFINE_PATH = "/home/ffajri/Workspace/RSTExtractor/models/biaffine"
BIAFFINE_MODEL = "network.pt"
RST_CONFIG_PATH = "/home/ffajri/Workspace/RSTExtractor/models/rst/config.cfg"
DATA_PATH = '/home/ffajri/Data/Petition/US/processed/merge/*'
THREADS = 10

if not os.path.exists('output_tree'):
    os.makedirs('output_tree')

def form_sentence(lines, word_alpha, char_alpha, tag_alpha, symbolic_root=False, symbolic_end=False):
    words = []
    word_ids = []
    seq_chars = []
    seq_char_ids = []
    tags = []
    tag_ids = []
    edu_ids = []
        
    if symbolic_root:
        words.append(ROOT)
        word_ids.append(word_alpha.get_index(ROOT))
        seq_chars.append([ROOT_CHAR, ])
        seq_char_ids.append([char_alpha.get_index(ROOT_CHAR), ])
        tags.append(ROOT_POS)
        tag_ids.append(tag_alpha.get_index(ROOT_POS))

    for line in lines:
        chars = []
        char_ids = []
        data = line.strip().split('\t')
        word = DIGIT_RE.sub(b"0", data[2])
        word_id = word_alpha.get_index(word)
        for c in words:
            chars.append(c)
            char_ids.append(char_alpha.get_index(c))
        tag = '$' if data[4] == '#' else data[4]
        tag_id = tag_alpha.get_index(tag)
        edu_id = int(data[9])

        words.append(word)
        word_ids.append(word_id)
        seq_chars.append(chars)
        seq_char_ids.append(char_ids)
        tags.append(tag)
        tag_ids.append(tag_id)
        edu_ids.append(edu_id)
    
    if symbolic_end:
        words.append(END)
        word_ids.append(word_alpha.get_index(END))
        seq_chars.append([END_CHAR, ])
        seq_char_ids.append([char_alpha.get_index(END_CHAR), ])
        tags.append(END_POS)
        tag_ids.append(tag_alpha.get_index(END_POS))
    return Sentence(words, seq_chars, tags, word_ids, seq_char_ids, tag_ids, edu_ids)

def data_reader(file_path, biaffine):
    f = open(file_path, 'r')
    sentences = []
    lines = []
    for line in f.readlines():
        if line.strip() == '':
            sentences.append(form_sentence(lines, biaffine.word_alpha, biaffine.char_alpha, biaffine.tag_alpha))
            lines = []
        else:
            lines.append(line)
    data = biaffine.prepare_data(sentences)
    syntax_features = biaffine.get_syntax_feature(data, sentences)
    
    for i in range(len(sentences)):
        assert(len(sentences[i].words) == syntax_features[i].shape[1])
    instance = Instance(sentences, syntax_features)
    return instance

files=glob.glob(DATA_PATH)

def run_thread(files):
    rst = RSTModel(RST_CONFIG_PATH)
    biaffine = BiaffineModel(BIAFFINE_PATH, BIAFFINE_MODEL)
    for filepath in files:
        filename = filepath.split('/')[-1].replace('.merge', '')
        instance = data_reader(filepath, biaffine)
        rst_data = rst.prepare_data([instance], 1)
        tree = rst.get_subtree(rst_data)[0]
        tree.save('output_tree/' + filename)

partitions  = []
size = int(math.ceil(1.0*len(files)/THREADS))
processes = list()
for i in range(THREADS):
    start = i * size
    end = start + size
    if end > len(files):
        end = len(files)
    p = files[start:end]
    
    process = Process(target=run_thread, args=(p,))
    process.start()
    processes.append(process)
    if end == len(files):
        break
for process in processes:
    process.join()
