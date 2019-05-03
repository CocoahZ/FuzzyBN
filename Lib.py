import numpy as np
import copy


class Node:
    def __init__(self, p=[]):
        self.prob = p

    @property
    def sum(self):
        return np.sum(self.prob)


class CalNode(Node):
    def __init__(self, p=[], cpt=[]):
        super().__init__(p)
        self.cpt = cpt


def cal_cpt(weight, terms):
    weight = weight[0]
    print(weight)
    print(len(weight))
    print(terms)
    cpt_shape = [terms for i in range(len(weight) + 1)]
    cpt_num = pow(terms, len(weight) + 1)
    mask_shape = [pow(terms, len(weight)), terms]
    cpt = np.zeros(cpt_num)
    mask = create_mask(mask_shape, terms)
    index = 0
    for i in range(terms):
        temp_mask = copy.copy(mask)
        for k in range(mask_shape[0]):
            for j in range(terms):
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
    return cpt


def create_mask(mask_shape, terms):
    mask = np.zeros(shape=mask_shape)
    for i in range(mask_shape[1]):
        value = 0
        j = 0
        while j < mask_shape[0]:
            for k in range(pow(mask_shape[1], i)):
                mask[j][terms-1-i] = value
                j += 1
            if value < 4:
                value += 1
            else:
                value = 0
    return mask
