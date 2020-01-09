import glob
import jieba


def get_content(path):
    with open(path,'r',encoding='gbk',errors='ignore') as f:
        content=''
        for l in f:
            l=l.strip()
            content+=l
        return content
def get_TF(words,topK=10):
    tf_dic ={}
    for w in words:
        tf_dic[w]=tf_dic.get(w,0)+1
    return sorted(tf_dic.items(),key=lambda x:x[1],reverse=True)[:topK]

def stop_word(path):
    with open(path) as f:
        return [l.strip() for l in f]

if __name__ == '__main__':
    files=glob.glob('./data/*.txt')
    corpus=[get_content(x) for x in files]
    index =1
    split_words =[x for x in jieba.cut(corpus[index]) if x not in stop_word('stop_words.txt')]
    print('content:'+corpus[index])
    print('分词效果：' +'/'.join(split_words))
    print('样本的top10:'+str(get_TF(split_words)))