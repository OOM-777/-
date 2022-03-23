# !/usr/bin/python3
# --encoding=utf-8--
import os
import jieba
from tensorflow import keras
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
import numpy as np
from config import *


def load_text(EMAIL):
    """
    处理用户输入的邮件内容为模型能识别的数据
    EMAIL (str)为邮件内容
    :return:(list) 格式化后的文本数据
    """
    test = jieba.cut(EMAIL)
    jieba_email = str(' '.join((test)))
    jieba_email = np.array(jieba_email)
    jieba_email = pd.DataFrame(jieba_email.reshape(1, -1))
    jieba_email = list(jieba_email.iloc[0].values.astype('str'))

    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(FEATURE_PATH, "rb")))
    # 加载TfidfTransformer
    tfidftransformer = pickle.load(open(TF_IDF_PATH, "rb"))
    test_tfidf = tfidftransformer.transform(loaded_vec.transform(jieba_email))

    svd = joblib.load(SVD_PATH)
    jieba_email = svd.transform(test_tfidf)
    jieba_email = jieba_email.reshape(1, -1)
    return jieba_email


# 贝叶斯
def bayers(EMAIL):
    jieba_email = load_text(EMAIL)
    new_model = joblib.load(filename=BAYERS_PATH)
    pre = int(new_model.predict(jieba_email))
    if pre == 1:
        return ("spam")
    elif pre == 0:
        return ("ham")


# KNN
def knn(EMAIL):
    jieba_email = load_text(EMAIL)
    new_model = joblib.load(filename=KNN_PATH)
    pre = int(new_model.predict(jieba_email))
    if pre == 1:
        return ("spam")
    elif pre == 0:
        return ("ham")


# 随机森林
def random_forest(EMAIL):
    jieba_email = load_text(EMAIL)
    new_model = joblib.load(filename=RANDOM_FOREST_PATH)
    pre = int(new_model.predict(jieba_email))
    if pre == 1:
        return ("spam")
    elif pre == 0:
        return ("ham")


# SVM
def svm(EMAIL):
    jieba_email = load_text(EMAIL)
    new_model = joblib.load(filename=SVM_PATH)
    pre = int(new_model.predict(jieba_email))
    if pre == 1:
        return ("spam")
    elif pre == 0:
        return ("ham")


# 决策树
def decision_tree(EMAIL):
    jieba_email = load_text(EMAIL)
    new_model = joblib.load(filename=DECSION_TREE_PATH)
    pre = int(new_model.predict(jieba_email))
    if pre == 1:
        return ("spam")
    elif pre == 0:
        return ("ham")


# CNN
def cnn(EMAIL):
    jieba_email = load_text(EMAIL)
    new_model = keras.models.load_model(CNN_PATH)
    jieba_email = jieba_email.reshape(1, 1, 20)
    pre = new_model.predict(jieba_email)
    test = list(pre[0])
    pre = test.index(max(test))
    if pre == 1:
        return ("spam")
    elif pre == 0:
        return ("ham")


# 逻辑回归
def logistic(EMAIL):
    jieba_email = load_text(EMAIL)
    new_model = joblib.load(filename=LOGISTIC_PATH)
    pre = int(new_model.predict(jieba_email))
    if pre == 1:
        return ("spam")
    elif pre == 0:
        return ("ham")


# 根据投票返回是否垃圾邮件
def Ensemble(web, EMAIL, flag=0):
    pre = []
    # 判断输入是否有效
    print(f"INFO: content:{EMAIL} flag: {flag}")  # test
    if len(web) == 0 or EMAIL.strip() == '':
        return "无效输入"

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if flag == 0:
        with open(f'{SAVE_PATH}save.txt', 'a', encoding='utf-8') as f:
            txt = EMAIL + '\n'
            f.writelines(txt)

    if 'knn' in web:
        temp = knn(EMAIL)
        pre.append(temp)

    if 'tree' in web:
        temp = decision_tree(EMAIL)
        pre.append(temp)

    if 'bayes' in web:
        temp = bayers(EMAIL)
        pre.append(temp)

    if 'svm' in web:
        temp = svm(EMAIL)
        pre.append(temp)

    if 'cnn' in web:
        temp = cnn(EMAIL)
        pre.append(temp)

    if 'forest' in web:
        temp = random_forest(EMAIL)
        pre.append(temp)

    if 'logi' in web:
        temp = logistic(EMAIL)
        pre.append(temp)

    print(f'INFO: models:{web}')
    print('INFO: predict:', pre)
    dict = {'ham': 0, 'spam': 0}

    # 遍历并累计各模型检测结果
    for key in pre:
        dict[key] = dict.get(key, 0) + 1

    print(f"INFO: result: {dict}")  # test
    if dict['spam'] > dict['ham']:
        return '垃圾邮件'
    else:
        return '正常邮件'