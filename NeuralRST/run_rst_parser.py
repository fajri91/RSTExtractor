import sys, time
import numpy as np
import random
from datetime import datetime

sys.path.append(".")

import argparse
import torch
import json

from in_out.reader import Reader
from in_out.util import load_embedding_dict, get_logger
from in_out.preprocess import create_alphabet
from in_out.preprocess import batch_data_variable
from models.vocab import Vocab
from models.metric import Metric
from models.config import Config
from models.architecture import MainArchitecture


main_path='/home/ffajri/'
def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', required=True)
    args = args_parser.parse_args()
    config = Config(None)
    config.load_config(args.config_path)
    
    logger = get_logger("RSTParser RUN", config.use_dynamic_oracle, config.model_path)
    word_alpha, tag_alpha, gold_action_alpha, action_label_alpha, etype_alpha = create_alphabet(None, config.alphabet_path, logger)
    vocab = Vocab(word_alpha, tag_alpha, etype_alpha, gold_action_alpha, action_label_alpha)
    
    network = MainArchitecture(vocab, config) 
    network.load_state_dict(torch.load(config.model_name))

    if config.use_gpu:
        network = network.cuda()
    network.eval()
    
    logger.info('Reading test instance')
    reader = Reader(config.test_path, config.test_syn_feat_path)
    test_instances  = reader.read_data()
    time_start = datetime.now()
    batch_size = config.batch_size
    span = Metric(); nuclear = Metric(); relation = Metric(); full = Metric()
    predictions = []
    total_data_test = len(test_instances)
    for i in range(0, total_data_test, batch_size):
        end_index = i+batch_size
        if end_index > total_data_test:
            end_index = total_data_test
        indices = np.array(range(i, end_index))
        subset_data_test = batch_data_variable(test_instances, indices, vocab, config)
        prediction_of_subtrees = network.loss(subset_data_test, None)
        predictions += prediction_of_subtrees
    for i in range(total_data_test):
        span, nuclear, relation, full = test_instances[i].evaluate(predictions[i], span, nuclear, relation, full)
    time_elapsed = datetime.now() - time_start
    m,s = divmod(time_elapsed.seconds, 60)
    logger.info('TEST is finished in {} mins {} secs'.format(m,s))
    logger.info("S: " + span.print_metric())
    logger.info("N: " + nuclear.print_metric())
    logger.info("R: " + relation.print_metric())
    logger.info("F: " + full.print_metric())



    import ipdb; ipdb.set_trace()
if __name__ == '__main__':
    main()
