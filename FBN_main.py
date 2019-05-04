import numpy as np
import Init
import Lib


def main():
    file_path = "Data//"
    node = []   # 存储节点的list
    terms, layer_num, net_shape = Init.init(file_path)  # 从文件中读取语言描述分级的个数，FBN网络层数，网络结构
    normalised_weights = Init.read_weight(file_path, net_shape)     # 从文件中读取节点权重
    input_prob = Init.read_input(file_path, net_shape)
    # print(normalised_weights)
    # print(len(input_prob[3]))
    for i in range(layer_num):
        layer_temp = []
        for k in net_shape[i]:
            temp = []
            for j in range(k):
                if i != layer_num - 1:      # 如果不是输入结点
                    temp.append(Lib.Node())
                else:   # 是中间需要计算的结点
                    temp.append(Lib.CalNode())
            layer_temp.append(temp)
        node.append(layer_temp)
    # print(len(net_shape[-1]))
    # print(len(node[-1][3]))
    # print(node)

    # k = 0
    # while k < len(net_shape[-1]):
    #     for i in net_shape[-1]:
    #         # print(i)
    #         for j in range(i):
    #             node[-1][k][j].prob = input_prob[k][j]
    #         k += 1

    Lib.load_input(net_shape, node[-1], input_prob)

    for i in range(len(node[0][0])):
        # print(normalised_weights[1][i])
        node[0][0][i].cpt = Lib.cal_cpt(normalised_weights[1][i], terms)    # 计算节点的CPT
    # for i in range(len(node[0][0])):
    #     node[0][0][i].joint = Lib.cal_joint_prob(node[-1][i], terms)  # 计算节点的父节点的联合概率分布
    #     node[0][0][i].prob = np.ones(terms)
    #     for k in range(terms):
    #         node[0][0][i].prob[k] = node[0][0][i].prob[k] * np.sum(node[0][0][i].cpt[k] * node[0][0][i].joint)
    #     # print(node[0][0][i].prob)
    Lib.cal_process(node[0][0], node[-1], terms)    # 计算节点的父节点的联合概率分布
    goal_node = Lib.CalNode()
    goal_node.cpt = Lib.cal_cpt(normalised_weights[0][0], terms)
    goal_node.joint = Lib.cal_joint_prob(node[0][0], terms)
    goal_node.prob = np.ones(terms)
    for i in range(terms):
        goal_node.prob[i] = goal_node.prob[i] * np.sum(goal_node.cpt[i] * goal_node.joint)
    print("goal: ", goal_node.prob)
    print(Lib.utility_value(goal_node, terms))
    # print(input_prob)
    Lib.sensitive_process(net_shape, node, terms, file_path, goal_node)
    Lib.load_input(net_shape, node[-1], input_prob)
    # print(node[-1][0][0].prob)
    Lib.load_input(net_shape, node[-1], input_prob, sensitive=True, num=0.1, node_id=[0, 0])
    # print(node[-1][0][0].prob)
    input_prob = Init.read_input(file_path, net_shape)
    Lib.load_input(net_shape, node[-1], input_prob, sensitive=True, num=0.2, node_id=[0, 0])
    # print(node[-1][0][0].prob)


if __name__ == '__main__':
    main()
