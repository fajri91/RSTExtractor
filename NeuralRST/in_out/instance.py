import numpy as np
from NeuralRST.in_out.node import Node

# representing one document / one set
class Instance(object):
    def __init__(self, total_words, total_tags, edus, gold_actions, result):
        self.total_words = total_words
        self.total_tags = total_tags
        self.edus = edus
        self.gold_actions = gold_actions
        self.result = result

    def evaluate(self, other_result, span, nuclear, relation, full): # is_trained=False, max_edu_size=0):
        main_subtrees = self.result.subtrees
        span.overall_label_count += len(main_subtrees)
        span.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].span_equal(main_subtrees[j]):
                    span.correct_label_count += 1
                    break
        
        nuclear.overall_label_count += len(main_subtrees)
        nuclear.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].nuclear_equal(main_subtrees[j]):
                    nuclear.correct_label_count += 1
                    break

        relation.overall_label_count += len(main_subtrees)
        relation.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].relation_equal(main_subtrees[j]):
                    relation.correct_label_count += 1
                    break

        full.overall_label_count += len(main_subtrees)
        full.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].full_equal(main_subtrees[j]):
                    full.correct_label_count += 1
                    break
        return span, nuclear, relation, full 

# representing 1 EDU
class EDU(object):
    def __init__(self, start_index, end_index):
        self.start_index = start_index # int
        self.end_index = end_index # int
        self.etype = '' # string
        self.words = [] # list of word (string)
        self.tags = [] # list of tag (string)
        self.syntax_features = []

# nuclear will be: NUCLEAR, SATELLITE, span
class SubTree(object):
    NUCLEAR='NUCLEAR'
    SATELLITE='SATELLITE'
    SPAN='span'

    def __init__(self):
        self.nuclear = ''
        self.relation = ''
        self.edu_start = -1
        self.edu_end = -1

    def clear(self):
        self.nuclear = ''
        self.relation = ''
        self.edu_start = -1
        self.edu_end = -1

    def span_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end
    
    def nuclear_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end and self.nuclear == tree.nuclear

    def relation_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end and self.relation == tree.relation

    def full_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end and self.relation == tree.relation and self.nuclear and tree.nuclear

    def get_str(self):
        return self.nuclear +' '+self.relation+' edu('+str(self.edu_start)+'-'+str(self.edu_end) +')'

class CResult(object):
    def __init__(self):
        self.subtrees = []
    
    def clear(self):
        self.subtrees = []
    
    def save(self, file_path):
        np.save(file_path, np.array(self.subtrees))

    def obtain_tree(self):
        p_subtree = {}
        subtrees = self.subtrees
        assert len(subtrees) % 2 == 0
        for idx in range(0, len(subtrees), 2):
            edu_span = (subtrees[idx].edu_start, subtrees[idx+1].edu_end)
            nuclear = subtrees[idx].nuclear + " " + subtrees[idx+1].nuclear
            relation = subtrees[idx].relation
            if 'span' == relation:
                relation = subtrees[idx+1].relation
            tree = Node(edu_span, nuclear, relation)
            
            #set child:
            if p_subtree.get(edu_span[0], None) is not None:
                tree.left = p_subtree[edu_span[0]]
                p_subtree[edu_span[0]].parent = tree
            elif subtrees[idx].edu_start == subtrees[idx].edu_end:
                leaf = Node((subtrees[idx].edu_start, subtrees[idx].edu_end), '', '')
                tree.left = leaf
                leaf.parent = tree
            if p_subtree.get(edu_span[1], None) is not None:
                tree.right = p_subtree[edu_span[1]]
                p_subtree[edu_span[1]].parent = tree
            elif subtrees[idx+1].edu_start == subtrees[idx+1].edu_end:
                leaf =  Node((subtrees[idx+1].edu_start, subtrees[idx+1].edu_end), '', '')
                tree.right = leaf
                leaf.parent = tree
            p_subtree[edu_span[0]] = tree
            p_subtree[edu_span[1]] = tree
        if len(subtrees) != 0:
            return p_subtree[0]
        else:
            return None

# representing ONE word
class SynFeat(object):
    def __init__(self, arc_dep, arc_head, rel_dep, rel_head):
        self.arc_dep = arc_dep
        self.arc_head = arc_head
        self.rel_dep = rel_dep
        self.rel_head = rel_head
        # self.lstm_out1 = lstm_out1
        # self.lstm_out2 = lstm_out2

    def concat(self):
        return self.arc_dep + self.rel_dep + self.arc_head + self.rel_head

