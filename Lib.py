import numpy as np
import copy
import Init


class Node:
    def __init__(self, p=[]):
        self.prob = p

    @property
    def sum(self):
        return np.sum(self.prob)


class CalNode(Node):
    def __init__(self, p=[], cpt=[], joint=[]):
        super().__init__(p)
        self.cpt = cpt
        self.joint = joint


def load_input(net_shape, node, input_prob, sensitive=False, num=0., node_id=[]):    # 加载输入
    k = 0
    while k < len(net_shape[-1]):
        for i in net_shape[-1]:
            for j in range(i):
                node[k][j].prob = input_prob[k][j]
            k += 1
    if sensitive:
        node[node_id[0]][node_id[1]].prob = process_input(node[node_id[0]][node_id[1]].prob, num)


def cal_cpt(weight, terms):     # 计算CPT
    # print(weight)
    # print(len(weight))
    # print(terms)
    cpt_shape = [terms for i in range(len(weight) + 1)]
    cpt_num = pow(terms, len(weight) + 1)
    mask_shape = [pow(terms, len(weight)), len(weight)]
    cpt = np.zeros(cpt_num)
    mask = create_mask(mask_shape, terms)
    index = 0
    for i in range(terms):
        temp_mask = copy.copy(mask)
        for k in range(mask_shape[0]):
            for j in range(len(weight)):
                if temp_mask[k][j] == i:
                    temp_mask[k][j] = 1
                else:
                    temp_mask[k][j] = 0
        for l in temp_mask:
            cpt[index] = np.dot(l, weight)
            index += 1
        # np.set_printoptions(threshold=1e+6)
        # print(i, temp_mask)
    cpt = cpt.reshape(cpt_shape)
    # print(cpt)
    return cpt


def cal_joint_prob(node, terms):    # 计算父节点的联合概率
    joint_prob = np.ones(pow(terms, len(node)))
    mask_shape = [pow(terms, len(node)), len(node)]
    mask = create_mask(mask_shape, terms)
    for i in range(len(joint_prob)):
        for j in range(len(node)):
            joint_prob[i] = joint_prob[i] * node[j].prob[int(mask[i][j])]
    joint_prob_shape = [terms for i in range(len(node))]
    joint_prob = joint_prob.reshape(joint_prob_shape)
    return joint_prob


def create_mask(mask_shape, terms):
    # print(mask_shape)
    mask = np.zeros(shape=mask_shape)
    for i in range(mask_shape[1]):
        value = 0
        j = 0
        while j < mask_shape[0]:
            for k in range(pow(terms, i)):
                mask[j][mask_shape[1]-1-i] = value
                j += 1
                if j >= mask_shape[0]:
                    break
            if value < terms - 1:
                value += 1
            else:
                value = 0
    # np.set_printoptions(threshold=1e+6)
    # print(mask)
    return mask


def cal_process(node, pre_node, terms):     # 计算节点的父节点的联合概率分布
    for i in range(len(node)):
        node[i].joint = cal_joint_prob(pre_node[i], terms)
        node[i].prob = np.ones(terms)
        for k in range(terms):
            node[i].prob[k] = node[i].prob[k] * np.sum(node[i].cpt[k] * node[i].joint)


def utility_value(output_node, terms):
    v_n = [i + 1 for i in range(terms)]
    u_v = []
    for i in range(terms):
        u_v.append((v_n[i] - v_n[0]) / (v_n[-1] - v_n[0]))
    d_dr = sum(output_node.prob * u_v)
    return d_dr


def process_input(prob, num):   # 对input data进行处理
    prob[0] = prob[0] + num
    for i in range(len(prob)):
        if prob[len(prob) - i - 1] != 0:
            prob[len(prob) - i - 1] -= num
            break
    return prob


def sensitive_process(net_shape, node, terms, file_path, goal_node):
    sensitive_num = np.array([0.0, 0.1, 0.2, 0.3])
    for i in range(len(sensitive_num)):
        print(sensitive_num[i])
        for k in range(len(net_shape[-1])):
            for j in range(net_shape[-1][k]):
                print("R", k, j, " Crisp value: ")
                input_prob = Init.read_input(file_path, net_shape)
                load_input(net_shape, node[-1], input_prob, sensitive=True, num=sensitive_num[i], node_id=[k, j])
                cal_process(node[0][0], node[-1], terms)
                goal_node.joint = cal_joint_prob(node[0][0], terms)
                goal_node.prob = np.ones(terms)
                for m in range(terms):
                    goal_node.prob[m] = goal_node.prob[m] * np.sum(goal_node.cpt[m] * goal_node.joint)
                print(utility_value(goal_node, terms))
