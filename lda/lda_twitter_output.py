import pandas as pd
import math
import glob
import pickle   # モデルを保存
import MeCab    # 形態素解析
import os
import re # 正規表現
import urllib.request
import unicodedata # 正規化用
import neologdn # 正規化用
import numpy as np
import tweepy # Twitter API
import demoji # 絵文字
from sklearn.decomposition import LatentDirichletAllocation as LDA # LDA Modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # BoW / Tfidf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------

tagger = MeCab.Tagger("/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")


# Tweepyを適用
auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN,ACCESS_SECRET)

api = tweepy.API(auth)

tweet_data = [] # 読み込んだツイートを格納

# ---------------------------------------------------------------

#ストップワードのインストール
def load_jp_stopwords(path="Japanese-revised.txt"):
    stopword_url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path, header=None)[0].tolist()

# 正規化その他処理
def normalization(text):
    avoid_patterns = ['【.{1,10}】']
    # 正規化
    text_normalized = neologdn.normalize(text)
    text_normalized = unicodedata.normalize('NFKC', text_normalized)

    # 数字と桁区切り文字を全て0に変換
    text_normalized = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text_normalized)
    text_normalized = re.sub(r'\d+', '0', text_normalized)

    # URLを削除
    text_normalized = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+','', text_normalized)

    # 絵文字を削除
    text_normalized = demoji.replace(string=text_normalized, repl='')

    # 不必要なパターンを削除
    for avoid_pattern in avoid_patterns:
        if re.search(avoid_pattern, text_normalized):
            text_normalized = re.sub(avoid_pattern, '', text_normalized)

    return text_normalized

# 形態素解析の処理部分
def preprocess_jp(series):
    # ストップワードの読み込み
    stop_words = load_jp_stopwords("Japanese-revised.txt")
    def tokenizer_func(text):
        tokens = []
        node = tagger.parseToNode(text)
        while node:
            features = node.feature.split(',')
            surface = features[6]
            if (surface == '*') or (len(surface) < 2) or (surface in stop_words):
                node = node.next
                continue
            noun_flag = (features[0] == '名詞')
            number_flag = (re.match(r'\d+', surface)) and (features[1] == '固有名詞')
            proper_noun_flag = (features[0] == '名詞') & (features[1] == '固有名詞')
            pronoun_flag= (features[1] == '代名詞')
            if proper_noun_flag and not number_flag:
                tokens.append(surface)
            elif noun_flag and not pronoun_flag:
                tokens.append(surface)
            node = node.next
        return " ".join(tokens)
    series = series.map(tokenizer_func)
    #---------------Normalization-----------#
    series = series.map(lambda x: x.lower())
    return series

#2人のユーザーの興味のcos類似度を計算
def similality(lda1, lda2):
    return np.dot(lda1, lda2) / (np.linalg.norm(lda1) * np.linalg.norm(lda2))

# 円グラフの表示
def roundgraph_save(ndarray, accountname):
  label = [str(n) for n in range(10)]
  fig=plt.figure()
  plt.title('@'+accountname)
  plt.pie(ndarray, labels=label, counterclock=False, startangle=90, autopct="%.1f%%")
  fig.savefig("pictures/" + accountname + "_lda_round.png")

# 特定のユーザーのジャンル確率を推定
def analyze(accountname):
    tweet_texts=[]

    for tweet in tweepy.Cursor(api.user_timeline,screen_name = accountname,exclude_replies = True).items():
        # RTとリプライは除外
        if ('RT' in tweet.text) and ('@' in tweet.text):
            pass 
        else:
            tweet_texts.append(normalization(str(tweet.text.rstrip())))
   # 前処理
    tweet_ss = preprocess_jp(pd.Series(tweet_texts))

    tweets=[]
    for stringdata in tweet_ss:
        tweets.append(stringdata)
        # テキストに文字列を保存
        with open('tweet_data_' + accountname + '.txt', 'a') as f:
            f.write(stringdata)

    X=vectorizer.transform(tweets)
    lda_model=loaded_model.transform(X)

    # 各トピックごとの分布割合を出力
    sum_lda = []
    for i,lda in enumerate(lda_model):
       if i == 0:
           sum_lda = lda
       else:
           sum_lda += lda
           sum_lda = sum_lda / np.linalg.norm(sum_lda)
    print('#'*10)
    topicid=[i for i, x in enumerate(lda) if x == max(lda)]
    print(lda," >>> topic",topicid)
    print("")

    # numpyの配列に変換
    average_ndarray = np.array(lda)
    # グラフを表示
    roundgraph_save(average_ndarray, accountname)
    #確率分布の保存
    ldaname= 'models/' + accountname + '_distribution.sav'
    pickle.dump(average_ndarray, open(ldaname, 'wb'))

    return average_ndarray

# ---------------------------------------------------------------

#LDAのロード
ldaname="models/lda.sav"
loaded_model = pickle.load(open(ldaname, 'rb'))

#tfidfのロード
Tfidfname= 'models/Tfidf.sav'
vectorizer=None
with open(Tfidfname, 'rb') as f:
    vectorizer=pickle.load(f)

#ジャンル確率の推定と円グラフの保存
accountname = ['hirox246', 'takapon_jp']
lda1=analyze(accountname[0])
lda2=analyze(accountname[1])

#類似度を出力
print("#"*10)
print('@'+accountname[0]+'さんと@'+accountname[1]+'さんの類似度は{}%です！！'.format(similality(lda1, lda2) * 100))
print("#"*10)
print('\n')