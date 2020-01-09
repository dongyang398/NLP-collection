# encoding:utf-8
import os
from os.path import isfile, join
import numpy as np
import re


# 获取词表，词向量
def get_wordlist_wordvectors():
    wordsList = np.load('wordsList.npy')  # 40000
    print('载入word列表')
    wordsList = wordsList.tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]
    wordVectors = np.load('wordVectors.npy')  # 40000*50
    print('载入文本向量')
    print(len(wordsList))
    print(wordVectors.shape)
    return wordsList, wordVectors


# 获取每个文件的单词数量大小
def get_word_nums():
    pos_files = ['pos/' + f for f in os.listdir('pos/') if isfile(join('pos/', f))]
    neg_files = ['neg/' + f for f in os.listdir('neg/') if isfile(join('neg/', f))]
    num_words = []
    for pf in pos_files:
        with open(pf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print('pos文件评价完结')
    for nf in neg_files:
        with open(nf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print('负面评价完结')

    num_files = len(num_words)
    print('文件总数', num_files)
    print('所有的词的数量', sum(num_words))
    print('平均文件词的长度', sum(num_words) / len(num_words))
    return num_words,num_files


# 剔除文本的特殊符号
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
num_dimensions = 300  # 每个词的向量纬度


def make_idsmatrix():
    max_seq_num = 250  # 时间序列长度
    num_words, num_files = get_word_nums()
    ids = np.zeros((num_files, max_seq_num), dtype='int32')
    file_count = 0
    for pf in pos_files:
        with open(pf, "r", encoding='utf-8') as f:
            indexCounter = 0
            line = f.readline()
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[file_count][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[file_count][indexCounter] = 399999  # 未知的词
                indexCounter = indexCounter + 1
                if indexCounter >= max_seq_num:
                    break
            file_count = file_count + 1

    for nf in neg_files:
        with open(nf, "r", encoding='utf-8') as f:
            indexCounter = 0
            line = f.readline()
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[file_count][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[file_count][indexCounter] = 399999  # 未知的词语
            indexCounter = indexCounter + 1
            if indexCounter >= max_seq_num:
                break
            file_count = file_count + 1
    np.save('idsMatrix', ids)


if __name__ == '__main__':
    make_idsmatrix()