# Scikit-learnを用いてLDAトピックモデリング
import pandas as pd
import glob
import pickle

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
#print(news_ss.head())

#形態素解析
import MeCab
tagger = MeCab.Tagger("-d ../../../usr/local/lib/mecab/dic/mecab-ipadic-neologd")
import os
import urllib.request
import unicodedata # 正規化用
import neologdn # 正規化用


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
    def tokenizer_func(text):
        tokens = []
        '''
        # 正規化
        text_normalized = neologdn.normalize(text)
        text_normalized = unicodedata.normalize('NFKC', text_normalized)

        # 数字と桁区切り文字を全て0に変換
        text_normalized = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text_normalized)
        text_normalized = re.sub(r'\d+', '0', text_normalized)
        '''

        node = tagger.parseToNode(str(text))
        while node:
            features = node.feature.split(',')
            surface = features[6]
            if (surface == '*') or (len(surface) < 2):
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

processed_news_ss = preprocess_jp(news_ss)
print(processed_news_ss)

# ストップワードの登録
stop_words = load_jp_stopwords("Japanese-revised.txt")

#BoWの作成
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vectorizer = CountVectorizer(stop_words=stop_words, max_df=.15)
count_data = count_vectorizer.fit_transform(processed_news_ss)
#Tfidfの作成
tfidf_vectorizer = TfidfTransformer()
tfidf_data = tfidf_vectorizer.fit_transform(count_data)
#print(tfidf_data.toarray())

#最適なモデル数の評価
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation as LDA

def gridsearch_best_model(tfidf_data, plot_enabled=True):
    # Define Search Param
    n_topics = [4,6,8,10,12,14]
    search_params = {'n_components': n_topics}
    # Init the Model
    lda = LDA(max_iter=25,               # Max learning iterations
              learning_method='batch',  
              random_state=0,            # Random state
              n_jobs = -1,               # Use all available CPUs)
              )
    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params, cv=5)
    # Do the Grid Search
    model.fit(tfidf_data)
    # Best Model
    best_lda_model = model.best_estimator_

    # ---------------------------------------------------------

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(tfidf_data))
    # Get Log Likelyhoods from Grid Search Output
    log_likelyhoods_score = [round(score) for score in model.cv_results_["mean_test_score"]]
    for scores in log_likelyhoods_score:
        print(scores)
    
    # --------------------------------------------------------

    return best_lda_model

best_lda_model = gridsearch_best_model(tfidf_data)

#分類結果の表示
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #", topic_idx, ":")
        long_string = ','.join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(long_string)
        #topic_wordcloud(topic_idx, fig, long_string)
number_words = 500 # トピックに出力する単語数
print_topics(best_lda_model, count_vectorizer, number_words)

#モデルの保存
filename = 'lda_sklearn_model.sav'
pickle.dump(best_lda_model, open(filename, 'wb'))

# BoW, Tfidfをcsvに出力
frame = {'CountVectorizer': count_vectorizer, 'Tfidf': tfidf_vectorizer}
df = pd.DataFrame(frame)
df.to_csv('learned_vector.csv')
