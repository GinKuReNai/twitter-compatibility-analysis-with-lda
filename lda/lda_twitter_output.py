import pandas as pd
import glob
import pickle   # モデルを保存
import MeCab    # 形態素解析
import os
import urllib.request
import unicodedata # 正規化用
import neologdn # 正規化用
import numpy as np
import tweepy
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation as LDA # LDA Modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # BoW / Tfidf


tagger = MeCab.Tagger("-Ochasen")

# Twitter API Key
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_SECRET = ""
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

# 形態素解析の処理部分
def preprocess_jp(series):
    # ストップワードの読み込み
    stop_words = load_jp_stopwords("Japanese-revised.txt")
    def tokenizer_func(text):
        tokens = []
        # 正規化
        text_normalized = neologdn.normalize(text)
        text_normalized = unicodedata.normalize('NFKC', text_normalized)

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
            pronoun_flag= (features[1] == '代名詞')
            if proper_noun_flag:
                tokens.append(surface)
            elif noun_flag and not pronoun_flag:
                tokens.append(surface)
            node = node.next
        return " ".join(tokens)
    series = series.map(tokenizer_func)
    #---------------Normalization-----------#
    series = series.map(lambda x: x.lower())
    return series

# -----------------------------------------------------------------

count_vectorizer = CountVectorizer()    # BoW
tfidf_vectorizer = TfidfTransformer()   # Tfidf

filename="lda_web_model.sav" # LDA-Model-saved filename

loaded_model = pickle.load(open(filename, 'rb')) # モデルの読み込み
# Tweetを取得して形態素解析
for tweet in tweepy.Cursor(api.user_timeline,screen_name = "hirox246",exclude_replies = True).items():
  test_data_ss = pd.Series(tweet.text)
processed_test_data_ss = preprocess_jp(test_data_ss)
print(processed_test_data_ss[::])
test_count_data = count_vectorizer.fit_transform(processed_test_data_ss) # BoWの生成
test_tfidf_data = tfidf_vectorizer.fit_transform(test_count_data)   # Tfidfの生成
doc_topic_mat = loaded_model.fit_transform(test_tfidf_data) # LDA Modelの生成
dominant_topic = np.argmax(doc_topic_mat, axis=1) # 最大値を最も適するトピックとして処理

test_data_df = pd.DataFrame(test_data_ss, columns=['text'])
test_data_df['topic_id'] = dominant_topic

# 各トピックごとの分布割合を出力
for i in test_data_df.index:
    print(dominant_topic[i])

# 最適トピック番号を出力
print(dominant_topic)