import random
import math
import pandas as pd
import sklearn.metrics as me

class NavieBayes:
    def __init__(self):
        pass

    @staticmethod
    def loadCsv(filename):
        """
        读入数据
        :filename:文件名
        :return:(list) 数据内容
        """
        dataset = pd.read_csv(filename, encoding='ANSI').values.tolist()
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        return dataset

    def splitDataset(self, filename, ration):
        """
        拆分数据集
        :filename:文件名
        :ration:划分比列
        :return:(list) 训练集 测试集
        """
        dataset = self.loadCsv(filename)
        trainSize = int(len(dataset) * ration)
        trainSet = []
        testSet = list(dataset)
        while len(trainSet) < trainSize:
            index = random.randrange(len(testSet))
            trainSet.append(testSet.pop(index))
        return [trainSet, testSet]

    def separateByClass(self, dataset):
        """
        基于类别划分数据
        :dataset:数据集
        :return:(dict) 按类别划分后的数据
        """
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if vector[-1] not in separated:
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    # 计算平均值
    def mean(self, nums):
        return sum(nums) / float(len(nums))

    # 计算标准差
    def stdev(self, nums):
        avg = self.mean(nums)
        # 计算方差
        variance = sum([pow(x - avg, 2) for x in nums]) / float(len(nums) - 1)
        return math.sqrt(variance)

    # 汇总数据集
    def summarize(self, dataset):
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries

    # 按类别汇总数据
    def summarizeByClass(self, dataset):
        separated = self.separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances)
        return summaries

    # 计算高斯概率密度函数
    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self, summaries, inputVector):

        """
        计算类别的概率
        :dataset:数据集
        :return:(dict) 按类别划分后的数据
        """
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        return probabilities

    def predict(self, summaries, inputVector):
        probabilities = self.calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def getPredictions(self, summaries, testSet):
        predictions = []
        testLabel = []
        for i in range(len(testSet)):
            testLabel.append(testSet[i][-1])
            result = self.predict(summaries, testSet[i])
            predictions.append(result)
        return predictions, testLabel

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0

if __name__ == "__main__":
    train = './train.csv'
    test = './test.csv'
    splitRatio = 0.67
    # 实例化对象
    bayes = NavieBayes()
    # 读入训练集
    trainingSet = bayes.loadCsv(train)
    # 读入测试集
    testSet = bayes.loadCsv(test)
    # 训练模型
    summaries = bayes.summarizeByClass(trainingSet)
    # 测试模型
    predictions, labels = bayes.getPredictions(summaries, testSet)
    # ----test----
    # ------------
    accuracy = bayes.getAccuracy(testSet, predictions)
    print(f'准确率: {accuracy}%')
    # ----使用sklearn计算各指标----
    # 精确率
    precision_score = me.precision_score(labels, predictions)
    # 召回率
    recall_score = me.recall_score(labels, predictions)
    # F1均值
    f1_score = me.f1_score(labels, predictions)
    # 混淆矩阵
    confusion_matrix = me.confusion_matrix(labels, predictions)
    # 输出结果
    print(f'精确率为:{precision_score:.5f}')
    print(f'召回率:{recall_score:.5f}')
    print(f'F1均值为:{f1_score:.5f}')
    print(confusion_matrix)