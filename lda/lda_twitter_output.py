import pandas as pd
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
#import emoji # 絵文字
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation as LDA # LDA Modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # BoW / Tfidf


tagger = MeCab.Tagger("/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

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
# urlを削除する処理の関数
'''
def remove_url(text):
    while re.search(r'(https?://[a-zA-Z0-9.-]*)', text):
        match = re.search(r'https?://[a-zA-Z0-9.-]*)',text)
        if match:
            replace = match.group(1).split('://')
            text = text.replace(match.group(1), replace[1])

# 絵文字を削除する処理の関数
def remove_emoji(src_str):
    return ''.join(c for c in src_str if c not in emoji.UNICODE_EMOJI)
'''

# 前処理
def pretreatment(text):
    # 正規化
    text_normalized = neologdn.normalize(text)
    text_normalized = unicodedata.normalize('NFKC', text_normalized)
    # 記号を削除
    code_regex = re.compile('[\t\s!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉'\
        '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    text_revised = code_regex.sub('', text_normalized)

    # 数字と桁区切り文字を全て0に変換
    text_revised = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text_revised)
    text_revised = re.sub(r'\d+', '0', text_revised)

    # URLの削除
    text_revised = re.sub('https?://[\da-zA-Z!\?/\+\-_~=;\.,\*&@#\$%\(\)\'\[\]]+', '', text_revised)
    #text_revised = remove_url(text_normalized)
    # 絵文字の削除
    #text_revised = ''.join(c for c in text_revised if c not in emoji.UNICODE_EMOJI)
    #text_revised = remove_emoji(text_revised)

    return text_revised


# 形態素解析の処理部分
def preprocess_jp(series):
    # ストップワードの読み込み
    stop_words = load_jp_stopwords("Japanese-revised.txt")
    def tokenizer_func(text):
        tokens = []
        # 前処理
        text_revised = pretreatment(text)
        # 形態素解析 textrevisedに注意
        node = tagger.parseToNode(str(text_revised))
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
tweet_texts = ''
# Tweetを取得して形態素解析
for tweet in tweepy.Cursor(api.user_timeline,screen_name = "hirox246",exclude_replies = True).items():
    # RTとリプライは除外
    if ('RT' in tweet.text) and ('@' in tweet.text):
        pass
    else:
        tweet_texts += tweet.text
        with open('tweet_data_pre.txt', 'a') as f:
            f.write(tweet.text)
test_data_ss = pd.Series(tweet_texts)

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
