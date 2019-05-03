import numpy as np
import Init
import Lib


def main():
    file_path = "Data//"
    node = []   # 存储节点的list
    terms, layer_num, net_shape = Init.init(file_path)  # 从文件中读取语言描述分级的个数，FBN网络层数，网络结构
    normalised_weights = Init.read_weight(file_path, net_shape)     # 从文件中读取节点权重
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
    print(node)
    print(net_shape)
    print(normalised_weights)
    print(np.shape(net_shape))
    Lib.cal_cpt(normalised_weights, terms)
    input_data = Init.read_input(file_path, net_shape)
    a = Lib.Node(input_data[0][0])
    b = Lib.CalNode()
    print(a.prob[1])
    print(a.sum)
    print(b.prob)
    print(b.sum)


if __name__ == '__main__':
    main()
