import os
from os.path import isfile, join
import numpy as np


# 获取词表，词向量
def get_wordlist_wordvectors():
    wordsList = np.load('wordsList.npy')
    print('载入word列表')
    wordsList = wordsList.tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]
    wordVectors = np.load('wordVectors.npy')
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
    return num_words

def analysic(num_words):
    a1=0
    a2=0
    a3=0
    a4=0
    a5=0
    for index in num_words:
        if index <= 100:
            a1+=1
        elif 200 >= index>100:
            a2+=1
        elif 300 >= index>200:
            a3+=1
        elif 400 >= index > 300:
            a4+=1
        else:
            a5+=1
    print(a1,a2,a3,a4,a5)


if __name__ == '__main__':
    num_words = get_word_nums()
    analysic(num_words)

