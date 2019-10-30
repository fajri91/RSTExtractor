import numpy as np

class RSTFeature (object):
    def __init__(self):
        self.ns_score = {}
        self.rel_type_score = {}
        # self.relations = ['attribution', 'background', 'cause', 'comparison', 'condition', 
                # 'contrast', 'elaboration', 'enablement', 'evaluation', 'explanation', 'joint',
                # 'mannermeans', 'summary', 'temporal', 'topic', 'sameunit', 'textualorganization', 'list']
        self.relations= ["purp", "cont", "attr", "evid", "comp", "list", "back", "same", "topic",
                "mann", "summ", "cond", "temp", "eval", "text", "cause", "prob", "elab"]

    def depth(self, node):
        left_depth = self.depth(node.left) if node.left else 0
        right_depth = self.depth(node.right) if node.right else 0
        return max(left_depth, right_depth) + 1

    def get_max_edu(self, node):
        if node.right is None:
            return node.edu_span[1]
        return self.get_max_edu(node.right)

    # author: Fajri Koto
    # 6 May 2019
    def generate_heuristic_feature(self, node):
        if node is None:
            print('WARNING: There is a None tree')
            return np.array([[0] * 21])

        # Initialization
        depth = self.depth(node)
        relation_score = {}
        for relation in self.relations:
            relation_score[relation] = 0
        max_score = 0
        for i in range(1,depth+1,1): max_score += i
        # Compute! Output is stored in self.ns_score and self.rel_type_score
        
        self.compute_ns_score(node, depth, depth)
        self.compute_relation_score(node, max_score, depth, relation_score)

        # Store output
        output = []
        vectors = []
        max_edu = self.get_max_edu(node)
        assert max_edu+1 == len(self.ns_score)
        for id_edu in range(max_edu+1):
            vector1 = self.ns_score[id_edu]
            vector2 = self.rel_type_score[id_edu]
            vectors.append(vector1+vector2)
        vectors = np.array(vectors, np.float32)
        return vectors

    # author: Fajri Koto
    # 6 May 2019
    def compute_ns_score(self, node, height, n_score):
        if node.left == None and node.right == None:
            assert node.edu_span[0] == node.edu_span[1]
            self.ns_score[node.edu_span[0]] = [1.0*n_score/height]
            return
        n1, n2 = node.nuclear.split(' ')
        if n1 == 'SATELLITE':
            self.compute_ns_score(node.left, height, n_score-1)
        else:
            self.compute_ns_score(node.left, height, n_score)
        if n2 == 'SATELLITE':
            self.compute_ns_score(node.right, height, n_score-1)
        else:
            self.compute_ns_score(node.right, height, n_score)

    #author Fajri Koto
    # 6 May 2019
    def compute_relation_score(self, node, max_score, depth, relation_score):
        if node.relation != '':
            if (relation_score.get(node.relation, -1) != -1):
                relation_score[node.relation]+=depth

        if node.left is None and node.right is None:
            assert node.edu_span[0] == node.edu_span[1]
            result = []
            
            # find if you are left or right
            if node.parent is not None:
                n1, n2 = node.parent.nuclear.split(' ')
                n1_v = 0; n2_v = 0
                if n1 == 'NUCLEAR':
                    n1_v = 1
                if n2 == 'NUCLEAR':
                    n2_v = 1

                if node.parent.left == node:
                    result.append(n1_v)
                    result.append(n2_v)
                else:
                    assert node.parent.right == node
                    result.append(n2_v)
                    result.append(n1_v)
            else:
                result.append(1)
                result.append(1)

            # Score of relations
            for relation in self.relations:
                result.append(1.0*relation_score[relation]/max_score)
            self.rel_type_score[node.edu_span[0]] = result
            return
        
        self.compute_relation_score(node.left, max_score, depth-1, relation_score.copy())
        self.compute_relation_score(node.right, max_score, depth-1, relation_score.copy())
