import gensim.models
import codecs
import numpy as np
import jieba

model_path = './data/zhiwiki_news.doc2vec'
start_alpha = 0.01
infer_epoch = 1000
docvec_size = 192


def simlarityCalu(vector1, vector2):
    vector1Mod = np.sqrt(vector1.dot(vector1))
    vector2Mod = np.sqrt(vector2.dot(vector2))
    if vector2Mod != 0 and vector1Mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1Mod * vector2Mod)
    else:
        simlarity = 0
    return simlarity


def doc2vec(file_name, model):
    doc = [w for x in codecs.open(file_name, 'r', 'utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all


if __name__ == '__main__':
    model = gensim.models.Doc2Vec.load(model_path)
    t1 = './data/T1.txt'
    t2 = './data/T2.txt'
    t1_doc2vec = doc2vec(t1, model)
    t2_doc2vec = doc2vec(t2, model)
    print(simlarityCalu(t1_doc2vec, t2_doc2vec))
