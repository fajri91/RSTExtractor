import random
from NeuralRST.transition.action import CAction
from NeuralRST.in_out.instance import SubTree

class Explorer(object):
    def __init__(self, vocab):
        self.gold_action_alpha = vocab.gold_action_alpha
        self.action_label_alpha = vocab.action_label_alpha

    def subtree_loss(self, subtree, gold_tree):
        subtree_size = len(gold_tree)
        loss = 3
        for i in range(subtree_size):
            gold_subtree = gold_tree[i]
            if subtree.span_equal(gold_subtree):
                loss -= 1
                if subtree.nuclear == gold_subtree.nuclear:
                    loss -= 1
                    if subtree.relation == gold_subtree.relation:
                        loss -= 1
                break
        return loss

    # CAction ac
    # Cstate error_cstate
    # SubTree[] gold_tree
    def nuclear_label_loss(self, ac, error_cstate, gold_tree):
        assert(error_cstate.stack_size >= 2)
        top0 = error_cstate.stack[error_cstate.stack_size - 1]
        top1 = error_cstate.stack[error_cstate.stack_size - 2]
        subtree0 = SubTree()
        subtree1 = SubTree()
        if ac.nuclear == CAction.NN:
            subtree0.edu_start = top0.edu_start
            subtree0.edu_end = top0.edu_end
            subtree0.nuclear = SubTree.NUCLEAR
            subtree0.relation = ac.label
            subtree1.edu_start = top1.edu_start
            subtree1.edu_end = top1.edu_end
            subtree1.nuclear = SubTree.NUCLEAR
            subtree1.relation = ac.label
        elif ac.nuclear == CAction.NS:
            subtree0.edu_start = top0.edu_start
            subtree0.edu_end = top0.edu_end
            subtree0.nuclear = SubTree.SATELLITE
            subtree0.relation = ac.label
            subtree1.edu_start = top1.edu_start
            subtree1.edu_end = top1.edu_end
            subtree1.nuclear = SubTree.NUCLEAR
            subtree1.relation = SubTree.SPAN
        elif ac.nuclear == CAction.SN:
            subtree0.edu_start = top0.edu_start
            subtree0.edu_end = top0.edu_end
            subtree0.nuclear = SubTree.NUCLEAR
            subtree0.relation = SubTree.SPAN
            subtree1.edu_start = top1.edu_start
            subtree1.edu_end = top1.edu_end
            subtree1.nuclear = SubTree.SATELLITE
            subtree1.relation = ac.label
        loss0 = self.subtree_loss(subtree0, gold_tree)
        loss1 = self.subtree_loss(subtree1, gold_tree)

        return loss0 + loss1


    def shift_loss(self, error_cstate, gold_tree):
        assert(error_cstate.stack_size >= 1)
        end = error_cstate.stack[error_cstate.stack_size - 1].edu_end
        gold_action_size = len(gold_tree)
        count = 0
        max_size = error_cstate.stack_size - 1
        for i in range(0, max_size):
            start = error_cstate.stack[i].edu_start
            for j in range(0, gold_action_size):
                gold_subtree = gold_tree[j]
                if start == gold_subtree.edu_start and end == gold_subtree.edu_end:
                    count += 1
        return count

    def reduce_loss(self,error_cstate, gold_tree):
        assert(error_cstate.stack_size >= 1)
        start = error_cstate.stack[error_cstate.stack_size - 1].edu_start
        gold_action_size = len(gold_tree)
        count = 0
        for i in range(error_cstate.next_index, error_cstate.edu_size):
            end = i
            for j in range(0, gold_action_size):
                gold_subtree = gold_tree[j]
                if start == gold_subtree.edu_start and end == gold_subtree.edu_end:
                    count += 1
        return count

    def get_reduce_candidate(self, error_cstate, gold_tree, candidate_actions):
        assert(error_cstate.stack_size >= 2)
        label_size = self.gold_action_alpha.size()
        tmp_acts = [] # 1 element is tuple (CAction, int)
        for nuclear in ['NN', 'NS', 'SN']:
            for label in self.action_label_alpha.alphas:
                ac = CAction(CAction.REDUCE, nuclear, label)
                action_str = ac.get_str()
                pad_id = self.gold_action_alpha.alpha2id['PAD']
                if self.gold_action_alpha.word2id(action_str) != pad_id:
                    loss = self.nuclear_label_loss(ac, error_cstate, gold_tree)
                    tmp_acts.append((ac, loss))
                    if loss == 0:
                        candidate_actions.append(ac)
                        return candidate_actions
        assert(len(tmp_acts) > 0)
        action_size = len(tmp_acts)
        min_loss = tmp_acts[0][1]
        min_index = 0
        for i in range(1, action_size):
            cur_iter = tmp_acts[i]
            cur_loss = cur_iter[1]
            if cur_loss < min_loss:
                min_index = i
                min_loss = cur_loss

        for i in range(action_size):
            cur_iter = tmp_acts[i]
            if cur_iter[1] == min_loss:
                candidate_actions.append(cur_iter[0])
        return candidate_actions

    # parameter:
    #  error_cstate (CState)
    #  gold_tree (SubTree [])
    # return CState optimal_action
    def get_oracle(self, error_cstate, gold_tree):
        candidate_actions = []
        ac = CAction('', '', '')
        if error_cstate.stack_size < 2:
            if error_cstate.next_index == error_cstate.edu_size:
                ac.set(CAction.POP_ROOT, '', '')
            else:
                ac.set(CAction.SHIFT, '', '')
            candidate_actions.append(ac)
        elif error_cstate.next_index == error_cstate.edu_size:
            ac.set(CAction.REDUCE, '', '')
        else:
            shift_loss = self.shift_loss(error_cstate, gold_tree)
            reduce_loss = self.reduce_loss(error_cstate, gold_tree)
            if shift_loss < reduce_loss:
                ac.set(CAction.SHIFT, '', '')
                candidate_actions.append(ac)
            elif shift_loss >= reduce_loss:
                ac.set(CAction.REDUCE, '', '')
                if shift_loss == reduce_loss:
                    shift_action = CAction(CAction.SHIFT, '', '')
                    candidate_actions.append(shift_action)
        if ac.is_reduce():
            candidate_actions = self.get_reduce_candidate(error_cstate, gold_tree, candidate_actions)
        minimum = 0
        maximum = len(candidate_actions)
        rand_index = int(random.random() * (maximum-minimum))
        # import ipdb; ipdb.set_trace()
        return candidate_actions[rand_index]
        
