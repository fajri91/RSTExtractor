import numpy as np

class Node(object):
    def __init__(self, edu_span, nuclear, relation):
        self.edu_span = edu_span
        self.nuclear = nuclear
        self.relation = relation
        self.left = None
        self.right = None
        self.parent = None

    def str(self):
        return self.nuclear + ' ' + self.relation

