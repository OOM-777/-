# coding:utf-8
# @Time:2021/1/317:27
# @Author : lijinlin
# @File:myTree.py
# @Software:PyCharm
import json
from math import log
import operator
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
import numpy as np

feature_path = './feature/new_data.pkl'
tfidftransformer_path = './feature/tfidftransformer.pkl'
SVD_PATH = './feature/svd.pkl'


# 计算信息熵
def calcShannonEnt(dataSet):
    """
    输入：数据集
    输出：数据集的香农熵
    描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 算概率
        shannonEnt -= prob * log(prob, 2)  # 算信息熵

    return shannonEnt


# 划分数据集
def splitDataSet(dataSet, axis, value):
    """
    输入：数据集，选择维度，选择值
    输出：划分数据集
    描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
    """

    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)

    return retDataSet


# 计算选择最好的
def chooseBestFeatureToSplit(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 等到总的特征
        # 去点重复值   获得每一列的值
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            # 第1列，值为value     拿到除了这一列以外其他列数据
            subDataSet = splitDataSet(dataSet, i, value)

            prob = len(subDataSet) / float(len(dataSet))

            newEntropy += prob * calcShannonEnt(subDataSet)

            splitInfo += -prob * log(prob, 2)
        # 信息增益
        infoGain = baseEntropy - newEntropy

        if (splitInfo == 0):  # fix the overflow bug  修复溢出错误
            continue
        # 信息增益率
        infoGainRatio = infoGain / splitInfo

        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio

            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1))
    return sortedClassCount[0][0]


# 总的架构创建树
def createTree(dataSet, labels):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 删除这一属性
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制所有标签，这样树就不会弄乱现有的标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = 0.0
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)

            else:
                classLabel = secondDict[key]

    return classLabel


def classifyAll(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


# 搞训练数据
def createDataSet():
    x_train = pd.read_csv('./data/train.csv')
    x_test = pd.read_csv('./data/test.csv')
    x_trains = x_train[
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18',
         '19']]  # 第9行数据有问题
    x_ = [i * 0.1 for i in range(-10, 10, 1)]
    for i in x_trains:
        x_trains[i] = pd.cut(x_trains[i], x_)
        x_trains[i] = x_trains[i].apply(lambda x: str(x))
    x_trains['label'] = x_train['label']
    y_label = x_test['label']
    x_test = x_test[
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]

    for i in x_test:
        x_test[i] = pd.cut(x_test[i], x_)
        x_test[i] = x_test[i].apply(lambda x: str(x))
    x_trains = x_trains.values.tolist()
    x_test = x_test.values.tolist()
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    return x_trains, labels, x_test, y_label


# 训练
def main():
    x_train, labels, x_test, y_label = createDataSet()
    labels_tmp = labels[:]  # 拷贝，createTree会改变labels
    desicionTree = createTree(x_train, labels_tmp)
    print('desicionTree:\n', desicionTree)
    js = json.dumps(desicionTree)
    file = open('./model/desicionTree.txt', 'w')
    file.write(js)
    file.close()
    print('正在预测，等待30秒：')
    y = classifyAll(desicionTree, labels, x_test)
    sum = 0
    for i in range(len(y)):

        if y[i] == y_label[i]:
            sum += 1

    print('classifyResult:\n', y)
    print('准确度：', sum / len(y))
    precision = precision_score(y_label, y)
    recall = recall_score(y_label, y)
    f1mean = f1_score(y_label, y)
    print('精确率为：%0.5f' % precision)
    print('召回率：%0.5f' % recall)
    print('F1均值为：%0.5f' % f1mean)


# 预测
def one_test():
    a = eval(input('检测邮件1，退出0'))
    if a == 1:
        email = input("请输入要检测的邮件：")
        test = jieba.cut(email)
        jieba_email = str(' '.join((test)))
        jieba_email = np.array(jieba_email)
        jieba_email = pd.DataFrame(jieba_email.reshape(1, -1))
        jieba_email = list(jieba_email.iloc[0].values.astype('str'))
        # 加载模型
        loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))
        # 加载TfidfTransformer
        tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))
        test_tfidf = tfidftransformer.transform(loaded_vec.transform(jieba_email))

        jieba_email = list(jieba_email)

        svd = joblib.load(SVD_PATH)
        jieba_email = svd.transform(test_tfidf)
        jieba_email = jieba_email.reshape(1, -1)

        x_ = [i * 0.1 for i in range(-10, 10, 1)]
        jieba_email = pd.DataFrame(jieba_email)
        jieba_email.drop(9, axis=1, inplace=True)

        for i in jieba_email:
            jieba_email[i] = pd.cut(jieba_email[i], x_)
            jieba_email[i] = jieba_email[i].apply(lambda x: str(x))

        jieba_email = jieba_email.values.tolist()
        print(jieba_email)
        file = open('./model/desicionTree.txt', 'r')
        js = file.read()
        desicionTree1 = json.loads(js)
        file.close()
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                  '19']
        pre = classifyAll(desicionTree1, labels, jieba_email)[0]
        if pre == 1:
            print("垃圾邮件")
        elif pre == 0:
            print("正常邮件")
        one_test()
    else:
        exit()


if __name__ == '__main__':
    main()  # 训练
    # one_test()#预测
