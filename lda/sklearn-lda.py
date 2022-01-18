# Scikit-learnを用いてLDAトピックモデリング
import pandas as pd # CSV I/O
import glob # パスを全探索
import pickle # モデル保存
import MeCab # 形態素解析
import os
import urllib.request
import unicodedata # 正規化用
import neologdn # 正規化用

# -------------------------------------------------------------------

# 探索パスのパターン
path_patterns = ['/home/s192c1058/data/*.txt', '/home/s192c1058/data/yomiuri/*.txt']
# 入力ファイル名
inputfiles = []
for path in path_patterns:
  inputfiles += glob.glob(path, recursive=False)

texts = []
#データの読み込み
for text_path in inputfiles:
    text = open(text_path, 'r').read()
    text = text.split('\n')
    #text = ' '.join(text[2:])
    texts.append(text)
news_ss = pd.Series(texts)

#形態素解析
tagger = MeCab.Tagger("-d ../../../usr/local/lib/mecab/dic/mecab-ipadic-neologd")

# ---------------------------------------------------------------------

#ストップワードのインストール
def load_jp_stopwords(path="Japanese-revised.txt"):
    stopword_url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    # ストップワードリストがない場合はURLからインストール
    else:
        print('Downloading...')
        urllib.request.urlretrieve(stopword_url, path)
    return pd.read_csv(path, header=None)[0].tolist()

# 形態素解析の処理部分
def preprocess_jp(series):
    # 前処理の処理部分
    def tokenizer_func(text):
        tokens = []
        # ストップワードの登録
        stop_words = load_jp_stopwords("Japanese-revised.txt")

        node = tagger.parseToNode(str(text))
        while node:
            features = node.feature.split(',')
            surface = features[6]
            if (surface == '*') or (len(surface) < 2) or (surface in stop_words):
                node = node.next
                continue
            noun_flag = (features[0] == '名詞')
            proper_noun_flag = (features[0] == '名詞') & (features[1] == '固有名詞')
            #location_flag= (features[2] == '地域')
            pronoun_flag = (features[1] == '代名詞')
            number_flag = (features[1] == '数') or (features[2] == '助数詞')
            # 名詞 かつ 固有名詞のときは追加
            if proper_noun_flag:
                tokens.append(surface)
            # 名詞 かつ 代名詞でないときは追加
            elif noun_flag and not pronoun_flag:
                tokens.append(surface)
            # 名詞 かつ 数 or 助数詞ではないときは追加
            elif noun_flag and not number_flag:
                tokens.append(surface)
            node = node.next
        return " ".join(tokens)

    series = series.map(tokenizer_func)
    #---------------Normalization-----------#
    series = series.map(lambda x: x.lower())
    return series

# ---------------------------------------------------------------------

#記事データに対して形態素解析
processed_news_ss = preprocess_jp(news_ss)

#訓練データの作成
traindata=[]
stringdata=''
for i in processed_news_ss:
    stringdata=i
    traindata.append(stringdata)

#BoWの作成(現在は省略)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#vect=CountVectorizer(max_features=10000,max_df=.50).fit(processed_news_ss)
#data=vect.transform(processed_news_ss)

#Tfidfの作成
tfidf_vec = TfidfVectorizer(lowercase=True, max_df = .50).fit(traindata)
X_train = tfidf_vec.transform(traindata)

# LDAモデルの作成
from sklearn.decomposition import LatentDirichletAllocation as LDA

topic_num=10
lda = LDA(n_components=topic_num,max_iter=25,               # Max learning iterations
              learning_method='batch',  
              random_state=0,            # Random state 
              )
lda.fit(X_train)

# 各トピックの上位100語を出力
sorting = np.argsort(lda.components_, axis=1)[:,::-1]
top_n=100
for i in range(topic_num):
    print("topic{}:".format(i))
    for j in range(top_n):
        print("ID: {0:4} Word: {1}".format(sorting[i][j],tfidf_vec.get_feature_names()[sorting[i][j]]))
    
    print()

#LDAモデルの保存
ldaname = 'lda.sav'
pickle.dump(lda, open(ldaname, 'wb'))

#Tfidfの保存
Tfidfname= 'Tfidf.sav'
pickle.dump(tfidf_vec, open(Tfidfname, 'wb'))