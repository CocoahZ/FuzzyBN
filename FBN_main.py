import numpy as np
import Init


file_path = "Data//"
net_shape = []    # 定义贝叶斯网络结点个数及结构
normalised_weights = []

layer_num, net_shape = Init.init(file_path)
normalised_weights = Init.read_weight(file_path, net_shape)
print(normalised_weights)
print(np.shape(normalised_weights))