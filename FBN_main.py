import numpy as np
import Init
import Lib


def main():
    file_path = "Data//"
    node = []   # 存储节点的list
    terms, layer_num, net_shape = Init.init(file_path)  # 从文件中读取语言描述分级的个数，FBN网络层数，网络结构
    normalised_weights = Init.read_weight(file_path, net_shape)     # 从文件中读取节点权重
    input_prob = Init.read_input(file_path, net_shape)
    # print(net_shape)
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
    k = 0
    while k < len(net_shape[-1]):
        for i in net_shape[-1]:
            # print(i)
            for j in range(i):
                node[-1][k][j].prob = input_prob[k][j]
            k += 1
    print(input_prob[0][0])
    # print(node[-1][4][2].prob)
    # print(node[-1][0][0].prob)
    # print(len(input_prob[0][0]))
    # print(net_shape)
    # print(normalised_weights)
    # print(np.shape(net_shape))
    cpt1 = Lib.cal_cpt(normalised_weights, terms)
    input_data = Init.read_input(file_path, net_shape)
    joint1 = Lib.cal_joint_prob(node[-1][1], terms)
    ans = [1, 1, 1, 1, 1]
    for i in range(len(ans)):
        ans[i] = ans[i] * np.sum(cpt1[i] * joint1)
    print(ans)
    a = Lib.Node(input_data[0][0])
    b = Lib.CalNode()
    # print(a.prob[1])
    # print(a.sum)
    # print(b.prob)
    # print(b.sum)


if __name__ == '__main__':
    main()
