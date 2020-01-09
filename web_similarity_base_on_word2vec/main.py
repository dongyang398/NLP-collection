# -*- coding: utf-8 -*-
import jieba.posseg as pseg
import analyse from jieba
import codecs
import numpy
import gensim
import numpy as np

# 提取关键字
def keyword_extract(data):
   tfidf = analyse.extract_tags
   keywords = tfidf(data)
   return keywords

#提取全文关键字
def getKeywords(docpath, savepath):
   with open(docpath, 'r') as docf, open(savepath, 'w') as outf:
      for data in docf:
         data = data[:len(data)-1]
         keywords = keyword_extract(data, savepath)
         for word in keywords:
            outf.write(word + ' ')
         outf.write('\n')

wordvec_size=192
def get_char_pos(string,char):
    chPos=[]
    try:
        chPos=list(((pos) for pos,val in enumerate(string) if(val == char)))
    except:
        pass
    return chPos

#用训练好的模型和关键字计算文本向量
def word2vec(file_name,model):
    with codecs.open(file_name, 'r') as f:
        word_vec_all = numpy.zeros(wordvec_size)
        for data in f:
            space_pos = get_char_pos(data, ' ')
            first_word=data[0:space_pos[0]]
            if model.__contains__(first_word):
                word_vec_all= word_vec_all+model[first_word]
            for i in range(len(space_pos) - 1):
                word = data[space_pos[i]:space_pos[i + 1]]
                if model.__contains__(word):
                    word_vec_all = word_vec_all+model[word]
        return word_vec_all

#计算相似度
def simlarityCalu(vector1,vector2):
    vector1Mod=np.sqrt(vector1.dot(vector1))
    vector2Mod=np.sqrt(vector2.dot(vector2))
    if vector2Mod!=0 and vector1Mod!=0:
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    return simlarity

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('data/zhiwiki_news.word2vec')
    t1 = './data/T1.txt'
    t2 = './data/T2.txt'
    t1_keywords = './data/T1_keywords.txt'
    t2_keywords = './data/T2_keywords.txt'
    getKeywords(t1, t1_keywords)
    getKeywords(t2, t2_keywords)
    p1_vec=word2vec(t1_keywords,model)
    p2_vec=word2vec(t2_keywords,model)
    print('T1文本和T2文本的相似度：'+simlarityCalu(p1_vec,p2_vec))