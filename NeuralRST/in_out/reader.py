import numpy as np
from multiprocessing import Queue as queue

from string import digits
from math import ceil

from NeuralRST.in_out.instance import Instance
from NeuralRST.in_out.instance import EDU
from NeuralRST.in_out.instance import SubTree
from NeuralRST.in_out.instance import CResult
from NeuralRST.in_out.instance import SynFeat
from NeuralRST.transition.action import CAction

# remove_digits = str.maketrans('', '', digits)

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_TAG = 1

CA_REDUCE = 'RD'
CA_SHIFT = 'SH'
CA_POP_ROOT = 'PR'
CA_NOACTION = ''

CA_NN = 'NN'
CA_NS = 'NS'
CA_SN = 'SN'

NULLKEY = ''

NUCLEAR = 'NUCLEAR'
SATELLITE = 'SATELLITE'
SPAN = 'span'


def modTupByIndex(tup, index, ins):
    lst = list(tup)
    lst[index] = ins
    return tuple(lst)

class Reader(object):
    def __init__(self, file_path, syntax_file_path):
        self.file_path = file_path
        self.syntax_file_path = syntax_file_path
    
    def parse_tree(self, string_tree, sent_types):
        # return value
        edus = []
        gold_actions = []
        
        subtree_stack = []
        op_stack = []
        relation_stack = []
        action_stack = []
        result = CResult()
        
        step = 0
        start = ''
        end = ''
        edu_start = 0
        edu_end = 0

        buffers = string_tree.split(' ')
        while(True):
            assert (step <= len(buffers))
            if step == len(buffers):
                break
            if buffers[step] == '(':
                op_stack.append(buffers[step])
                relation_stack.append(buffers[step+1])
                action_stack.append(buffers[step+2])
                if buffers[step + 2] == 't':
                    start = buffers[step + 3]
                    end = buffers[step + 4]
                    step += 2
                step += 3
            elif buffers[step] == ")":
                action = action_stack[-1] #stack.top
                if action == 't':
                    edu = EDU(int(start), int(end))
                    for j in range(len(sent_types)):
                        if edu.start_index >= sent_types[j][0] and edu.end_index <= sent_types[j][1]:
                            edu.etype = sent_types[j][2]
                            break
                    edu_start = len(edus)
                    edu_end = len(edus)
                    subtree_stack.append((edu_start, edu_end))
                    edus.append(edu)
                    ac = CAction(CA_SHIFT, '', relation_stack[-1])
                    assert(relation_stack[-1] == 'leaf')
                    gold_actions.append(ac)
                elif action == 'l' or action == 'r' or action == 'c':
                    nuclear = ''
                    if action == 'l':
                        nuclear = CA_NS
                    elif action == 'r':
                        nuclear = CA_SN
                    elif action == 'c':
                        nuclear = CA_NN
                    ac = CAction(CA_REDUCE, nuclear, relation_stack[-1])
                    gold_actions.append(ac)

                    subtree_size = len(subtree_stack)
                    assert(subtree_size >= 2)

                    right_tree_index = subtree_stack[subtree_size-1]
                    left_tree_index = subtree_stack[subtree_size-2]
                    
                    right_tree = SubTree()
                    right_tree.edu_start = right_tree_index[0]
                    right_tree.edu_end = right_tree_index[1]
                    
                    left_tree = SubTree()
                    left_tree.edu_start = left_tree_index[0]
                    left_tree.edu_end = left_tree_index[1]

                    if action == 'l':
                        left_tree.nuclear = NUCLEAR
                        right_tree.nuclear = SATELLITE
                        left_tree.relation = SPAN
                        right_tree.relation = ac.label
                    elif action == 'r':
                        left_tree.nuclear = SATELLITE
                        right_tree.nuclear = NUCLEAR
                        left_tree.relation = ac.label
                        right_tree.relation = SPAN
                    elif action == 'c':
                        left_tree.nuclear = NUCLEAR
                        right_tree.nuclear = NUCLEAR
                        left_tree.relation = ac.label
                        right_tree.relation = ac.label
                    
                    result.subtrees.append(left_tree)
                    result.subtrees.append(right_tree)

                    edu_start = right_tree_index[0]
                    edu_end = right_tree_index[1]
                    assert(left_tree_index[1] + 1 == edu_start)
                    subtree_stack[subtree_size-2] = modTupByIndex(left_tree_index, 1, edu_end)
                    subtree_stack.pop()
                
                action_stack.pop()
                relation_stack.pop()
                op_stack.pop()
                step += 1

        ac = CAction(CA_POP_ROOT, '', '')
        gold_actions.append(ac)
        #Check stack
        assert(len(op_stack) == 0 and len(relation_stack)==0 and  len(action_stack)==0)
        return edus, gold_actions, result

    def process_instance(self, subline):
        words = []
        tags = []
        total_words = []
        total_tags = []
        gold_tree = ''
        sent_types = []

        if len(subline) == 0:
            return None
        if len (subline) % 2 == 0:
            raise Exception('Each sublines in document input must not be even')

        start_index = 0
        end_index = 0
        for i in range(int(len(subline)/2)):
            word_tag_pairs = subline[i].split(' ')
            end_index = start_index + len(word_tag_pairs) - 1
            for word_tag_pair in word_tag_pairs:
                if word_tag_pair == '<P>' or word_tag_pair == '<S>':
                    break
                try:
                    word, tag = word_tag_pair.split('_')
                except:
                    print (word_tag_pairs)

                words.append(word.lower())
                tags.append(tag)
                total_words.append(word)
                total_tags.append(tag)
            total_words.append(word_tag_pairs[-1])
            total_tags.append(NULLKEY)
            sent_types.append((start_index, end_index, word_tag_pair))
            start_index = end_index + 1
        edus, gold_actions, result = self.parse_tree(subline[-1], sent_types)

        edu_size = len(edus)
        total_word_size = len(total_words)
        total_tag_size = len(total_tags)
        type_size = len(sent_types)
        word_size = len(words)
        
        # Check word and tag num
        assert(total_tag_size == total_word_size)
        assert(word_size + type_size == total_tag_size)
        sum = 0
        # Check edu
        for i in range(edu_size):
            assert(edus[i].start_index <= edus[i].end_index)
            assert(edus[i].start_index >= 0 and edus[i].end_index < total_word_size)
            if (i < edu_size - 1):
                assert(edus[i].end_index + 1 == edus[i+1].start_index)
            for j in range(edus[i].start_index, edus[i].end_index + 1):
                if total_tags[j] != NULLKEY:
                    edus[i].words.append(total_words[j].lower())
                    edus[i].tags.append(total_tags[j])
                else:
                    sum+=1
            assert(len(edus[i].words) == len(edus[i].tags))
            sum += len(edus[i].words)
        
        #Check subtree
        assert(sum == len(total_words) and len(result.subtrees) + 2 == len(edus) * 2)
        subtree_size = len(result.subtrees)
        for i in range(subtree_size):
            assert(result.subtrees[i].relation != NULLKEY and result.subtrees[i].nuclear != NULLKEY)

        instance = Instance(total_words, total_tags, edus, gold_actions, result)
        return instance

    def read_main_data(self):
        instances = []
        f = open(self.file_path, 'r')
        subline = []
        for line in f.readlines():
            line = line.strip()
            if line == '':
                instance = self.process_instance(subline)
                instances.append(instance)
                subline = []
            else:
               subline.append(line)
        return np.array(instances)
    
    def read_dependency_feature(self, instances):
        idx = 0
        sum_word = 0
        f = open(self.dependency_file_path)
        depFeat = DepFeat()
        for line in f.readlines():
            line = line.strip()
            if line == '': # 1 sentence
                instances[idx].dep_features.append(depFeat)
                depFeat = DepFeat()
                if len(instances[idx].sent_types) == len(instances[idx].dep_features):
                    idx += 1
            else:
                data = line.split('\t')
                depFeat.words.append(data[1].lower())
                depFeat.tags.append(data[3])
                depFeat.dep_relations.append(data[7])
           
        #Check the whole result whether each words is inline or no
        for idx in range (len(instances)):
            word_num = 0
            for j in range(len(instances[idx].dep_features)):
                word_num += len(instances[idx].dep_features[j].words)
            assert (word_num == len(instances[idx].words))
            i = 0; offset = 0
            for j in range(word_num):
                assert(instances[idx].words[j] == instances[idx].dep_features[i].words[j - offset] and \
                        instances[idx].tags[j] == instances[idx].dep_features[i].tags[j - offset])
                if (j - offset + 1 == len(instances[idx].dep_features[i].words)):
                    offset += len(instances[idx].dep_features[i].words)
                    i += 1

        return instances
   
    def read_syntax_feature(self, instances):
        file1 = self.syntax_file_path + '/arc_dep'
        file2 = self.syntax_file_path + '/rel_head'
        file3 = self.syntax_file_path + '/arc_head'
        file4 = self.syntax_file_path + '/rel_dep'
        file5 = self.syntax_file_path + '/lstm_out'

        r1 = open(file1, 'r')
        r2 = open(file2, 'r')
        r3 = open(file3, 'r')
        r4 = open(file4, 'r')
        r5 = open(file5, 'r')

        def process_syn_feat(string, size):
            data = string.split(' ')
            ret_value = []
            assert(len(data) == size + 1)
            for i in range (1, size+1):
                ret_value.append(float(data[i]))
            return data[0], ret_value

        inst_idx = 0; edu_idx = 0; word_idx = 0; cur_word_size = 0
        while(True):
            str1 = []; str2 = []; str3 = []; str4 = []; str5 = []

            while(True):
                in1 = r1.readline().strip()
                in2 = r2.readline().strip()
                in3 = r3.readline().strip()
                in4 = r4.readline().strip()
                in5 = r5.readline().strip()

                if in1 != '':
                    assert(in2 != '' and in3 != '' and in4 != '' and in5 != '')
                    str1.append(in1)
                    str2.append(in2)
                    str3.append(in3)
                    str4.append(in4)
                    str5.append(in5)
                else:
                    assert(in2 == '' and in3 == '' and in4 == '' and in5 == '')
                    break
            if len(str1) == 0:
                if inst_idx != len(instances):
                    raise Exception ("Some instances do not have external features: " + str(inst_idx) + ":" + str(len(instances)) + '\n')
                break
            if len(str2) != len(str1) or len(str3) != len(str1) or len(str4) != len(str1) or len(str5) != len(str1):
                raise Exception ("Extern feature input error, vectors dont have the same length\n")
        
            for i in range(len(str1)):
                w1, arc_dep = process_syn_feat(str1[i], 500)
                w2, rel_head = process_syn_feat(str2[i], 100)
                w3, arc_head = process_syn_feat(str3[i], 500)
                w4, rel_dep = process_syn_feat(str4[i], 100)
                # total size of lstm_out = 800, but its splitted into two variable in the original implementation
                # lstm_out = process_syn_feat(str5[i], 800)
                # lstm_out1 = lstm_out[0:400]
                # lstm_out2 = lstm_out[400:800]
                assert (w1 == w2 and w1 == w3 and w1 == w4)
                syn_feat = SynFeat(arc_dep, arc_head, rel_dep, rel_head)
                try:
                    # a minor mistake in BiAffine implicit syntax feature, all digits are changed into 0
                    # The original data still exist in the main set. For the sake of easy test, I remove digits in words.
                    word1 = w1.lower().translate(None, digits)
                    word2 = instances[inst_idx].edus[edu_idx].words[cur_word_size].translate(None, digits)
                    assert (word1 == word2)
                except:
                    print('Tada!, Investigate why :3 cemungudh..')
                    import ipdb; ipdb.set_trace()
                
                instances[inst_idx].edus[edu_idx].syntax_features.append(syn_feat.concat())
                cur_word_size += 1
                if cur_word_size == len(instances[inst_idx].edus[edu_idx].words):
                    cur_word_size = 0
                    edu_idx += 1
                    if edu_idx == len(instances[inst_idx].edus):
                        edu_idx = 0
                        inst_idx += 1
        return instances

    def read_data(self):
        instances = self.read_main_data()
        instances = self.read_syntax_feature(instances)
        return instances
