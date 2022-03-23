import numpy as np
import operator
import time
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


# KNN
def KNN(x_test, trainset, trainlabel, k):
    trainset_size = trainset.shape[0]
    # 计算欧式距离
    # tile函数用于复制单条测试集
    den_test = np.tile(x_test, (trainset_size, 1)) - trainset
    temp = den_test ** 2
    disences = temp.sum(axis=1)
    disences = disences ** 0.5
    # 对距离值进行排序
    sort_Dis = disences.argsort()
    classCount = {}
    for i in range(k):

        test_lable = trainlabel[sort_Dis[i]]
        classCount[test_lable] = classCount.get(test_lable, 0) + 1
    sort_count = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sort_count[0][0]



train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

x_train = train.iloc[:, :20]
y_train = train.iloc[:, [20]]

x_test = test.iloc[:, :20]
y_test = test.iloc[:, [20]]

y_train = list(y_train['label'].astype('int'))
y_test = list(y_test['label'].astype('int'))

errorCount = 0.0

x_test = np.array(x_test)
y_test = np.array(y_test)
i = 0
dict = {1: '垃圾邮件', 0: '正常邮件'}
start = time.time()
y_pre = []
for temp in x_test:
    result = KNN(temp, x_train, y_train, 5)
    y_pre.append(result)
    print("KNN识别为: %s,真实为: %s" % (dict[result], dict[y_test[i]]))
    if (result != y_test[i]):
        errorCount += 1.0
    i += 1
end = time.time()
print("执行完毕，耗时%.2fs" % (end - start))

precision = precision_score(y_test, y_pre)
recall = recall_score(y_test, y_pre)
f1mean = f1_score(y_test, y_pre)
print("==============KNN==============")
print("准确率为：%f" % (1 - (errorCount / len(y_test))))
print('精确率为：%0.5f' % precision)
print('召回率为：%0.5f' % recall)
print('F1均值为：%0.5f' % f1mean)