import csv
import numpy


def init(file_path):
    i = 0
    layer_num = 0
    ans = []
    file_object = open(file_path + 'NetShape.ini')
    for line in file_object.readlines():
        cur_line = line.strip().split("\t")
        if cur_line[0] == 'Layer_num':
            layer_num = int(cur_line[1])
            i += 1
        # elif i <= layer_num:
        #     print(cur_line[0])
        elif (i <= layer_num) and (cur_line[0] == 'Layer_Node_' + str(i)):
            temp = [int(j) for j in cur_line[1:]]
            ans.append(temp)
            i += 1
    return layer_num, ans


def read_weight(file_path, net_shape):
    file_object = open(file_path + 'Normalised_Weights.CSV')
    context = []
    ans = []
    layer_num = len(net_shape)
    j = 0
    with file_object as f:
        reader = csv.reader(f)
        for i in reader:
            context.append(i)
    for i in range(layer_num):
        for k in net_shape[i]:
            temp = []
            for l in range(k):
                temp.append(float(context[j][1]))
                j += 1
            ans.append(temp)
    return ans


