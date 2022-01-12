import pandas as pd
import glob
import pickle   # モデルを保存
import MeCab    # 形態素解析
import os
import urllib.request
import unicodedata # 正規化用
import neologdn # 正規化用
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation as LDA # LDA Modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # BoW / Tfidf
from sklearn.metrics.pairwise import euclidean_distances


tagger = MeCab.Tagger("-Ochasen")

# ---------------------------------------------------------------

#ストップワードのインストール
def load_jp_stopwords(path="Japanese-revised.txt"):
    stopword_url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        urllib.request.urlretrieve(stopword_url, path)
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

# 推定したいテキストに対してトピックを推定するメソッド
def predict_topic(text, doc_topic_probs, top_n=5):
    # 前処理
    processed_test_data_ss = preprocess_jp(pd.Series(test_data_ss))
    # BoW, Tfidfの作成
    count_vectorizer = CountVectorizer()    # BoW
    tfidf_vectorizer = TfidfTransformer()   # Tfidf

    # それぞれfit_transform
    count_data = count_vectorizer.fit_transform(processed_test_data_ss) # BoWの生成
    tfidf_data = tfidf_vectorizer.fit_transform(count_data)   # Tfidfの生成

    # トピックの確率分布の推定
    topic_probability_scores = loaded_model.transform(tfidf_data)
    dists = euclidean_distances(topic_probability_scores.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = np.argsort(dists)[:top_n]
    # 確率分布
    topic_prob_scores = np.round(topic_probability_scores, 1)
    most_similar_docprobs = np.round(doc_topic_probs[doc_ids], 1)

    return topic_prob_scores, most_similar_docprobs

# -----------------------------------------------------------------

# モデルの読み込み
filename="/home/s192c2107/lda_web_model.sav" # LDA-Model-saved filename
loaded_model = pickle.load(open(filename, 'rb')) # モデルの読み込み

# BoW, Tfidfの読み込み
df = pd.read_csv('learned_vector.csv')
cv_s = df['CountVectorizer']
tf_s = df['Tfidf']
count_vectorizer = cv_s.values
tfidf_vectorizer = tf_s.values

# ------------------------------------------------------------------

test_data_ss = ['東京・池袋で２０１９年４月、母子２人が死亡、９人が重軽傷を負った暴走事故で、自動車運転死傷行為処罰法違反（過失運転致死傷）に問われた旧通産省工業技術院の元院長・飯塚幸三被告（９０）を禁錮５年（求刑・禁錮７年）の実刑とした東京地裁判決が１７日、確定した。２日の判決後、被告側と検察側の双方が期限の１６日までに控訴しなかった。検察当局は被告を刑務所に収容する手続きに入るが、年齢や体調などを考慮し、刑の執行を停止する可能性もある。判決によると、被告は１９年４月１９日、豊島区東池袋で乗用車を運転中、ブレーキとアクセルを踏み間違えて暴走。主婦の松永真菜さん（当時３１歳）と長女の 莉子りこ ちゃん（同３歳）を時速約９６キロではねて死亡させ、通行人ら９人に重軽傷を負わせた。']
print(processed_test_data_ss[::])

# LDA Modelの生成
lda_output = loaded_model.transform(tfidf_vectorizer)

#確率分布を算出 
topic_vector1, topic_vector2 = predict_topic(test_data_ss, lda_output)
print(vector1)
print(vector2)

'''
# 最大値を最も適するトピックとして処理
dominant_topic = np.argmax(doc_topic_mat, axis=1)

test_data_df = pd.DataFrame(test_data_ss, columns=['text'])
test_data_df['topic_id'] = dominant_topic

pd.set_option('display.max_rows', None)
print(test_data_df)

# 最適なトピック
for i in test_data_df.index:
    print(dominant_topic[i])

# 各トピックごとの分布割当
print(doc_topic_mat)
'''