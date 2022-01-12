# GensimでLDAトピックモデリング
from collections import defaultdict
from gensim import models, corpora # Gensim Library
import unicodedata # 正規化用
import neologdn # 正規化用
import glob # パスの一括取得
import sys
import MeCab # 形態素解析
import re # 正規表現
import pandas # CSV I/O

# ------------------------------------------------------

# 探索パスのパターン
#aozora_path_pattern = '../aozorabunko_text/cards/*/files/*_ruby_*/*_utf-8_revised.txt'
path_patterns = ['/home/s192c1058/data/*.txt', '/home/s192c1058/data/yomiuri/*.txt']
# 入力ファイル名
inputfile = []
for path in path_patterns:
  inputfile += glob.glob(path, recursive=False)
# MeCabの初期化
#mc = MeCab.Tagger("/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
mc = MeCab.Tagger("-d ../../../usr/local/lib/mecab/dic/mecab-ipadic-neologd")
mc.parse('')

# ストップワードリスト
stopwords_file = open('Japanese-revised.txt', 'r')
stopwords = [line.strip() for line in stopwords_file]
stopwords = [ss for ss in stopwords if not ss==u'']
stopwords_file.close()
# 形態素解析結果
texts = []

# ---------------------------------------------------

# 形態素解析メソッド
def get_tokens(sent):
  # 正規化
  sent_normalized = neologdn.normalize(sent)
  sent_normalized = unicodedata.normalize('NFKC', sent_normalized)

  # 数字と桁区切り文字を全て0に変換
  sent_normalized = re.sub(r'(\d)([,.])(\d+)', r'\1\3', sent_normalized)
  sent_normalized = re.sub(r'\d+', '0', sent_normalized)

  node = mc.parseToNode(sent_normalized)
  tokens = []
  # 固有名詞を含めて代名詞を除く
  while node:
    features = node.feature.split(',')
    surface = features[6]
    if (surface == '*') or (len(surface) < 2) or (surface in stopwords):
      node = node.next
      continue
    noun_flag = (features[0] == '名詞')
    proper_noun_flag = (features[0] == '名詞') & (features[1] == '固有名詞')
    pronoun_flag= (features[1] == '代名詞')
    if proper_noun_flag:
      tokens.append(surface)
    elif noun_flag and not pronoun_flag:
      tokens.append(surface)
    node = node.next

  return tokens

# ストップワードを削除するメソッド
def del_stopwords(sent, stopwords=[]):
  revised_texts = []  # 削除後のテキスト
  for text in sent:
    for word in text:
      if not word in stopwords:
        texts.append(word)

  return revised_texts

# LDA結果のトピックを出力するメソッド
def lda_print_topics(lda):
  for i in range(N):
    print("\n")
    print("="*80)
    print("TOPIC {0}\n".format(i))
    topic = lda.show_topic(i)
    for t in topic:
        print("{0:20s}{1}".format(t[0], t[1]))

# モデルの評価指標を算出するメソッド
def compute_metrics(model, texts, corpus, dct):
  # Compute Perplexity(モデルの予測精度の評価指標)
  perplexity = lda.log_perplexity(corpus)

  # Compute Coherence Score（トピックの品質を測る評価指標）
  coherence_model_lda = models.CoherenceModel(model=model, texts=texts, dictionary=dct, coherence='u_mass')
  coherence_lda = coherence_model_lda.get_coherence()

  return perplexity, coherence_lda


# -------------------------------------------------
          
# 形態素解析した結果をtextsに追加
for i in range(len(inputfile)):
  with open(inputfile[i], 'r') as f:
    for line in f:
      texts.append(get_tokens(line))

# ストップワードを削除
texts = del_stopwords(texts, stopwords)

# 形態素解析の結果を出力
with open('neologd_outcome.txt','w') as f:
  f.write(str(texts))

# 単語の頻度の辞書
frequency = defaultdict(int)
for text in texts:
  for token in text:
    frequency[token] += 1


# n回以上出現する単語に限定
n = 10
processed_corpus = [[token for token in text if frequency[token] > n] for text in texts]


# 単語ごとにIDを割り振る
dictionary = corpora.Dictionary(processed_corpus)
dictionary.filter_extremes(no_above=0.3)
dictionary.save_as_text('/home/s192c1058/models/dict.txt')

# BoWを作成
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

# Tfidfを作成
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

perplexity = []
coherence = []
# トピック数を5ずつ増やしてLDAモデルを作成
for N in range(5, 21, 5):  
  # LDAを作成
  lda = models.ldamodel.LdaModel(corpus_tfidf, num_topics=N, id2word=dictionary)
  #corpus_lda = lda_model[corpus_tfidf]

  # モデルを保存
  lda.save(f'/home/s192c1058/models/models_{N}.model')

  # トピックを出力
  lda_print_topics(lda)

  # 評価指標を算出
  pp, ch = compute_metrics(model=lda, texts=processed_corpus, corpus=corpus_tfidf, dct=dictionary)
  perplexity.append(pp)
  coherence.append(ch)

  with open('ppch.txt', 'w') as f:
    f.write(pp)
    f.write(ch)

# 評価指標をcsvに出力
frame = {'perplexity': perplexity, 'coherence': coherence}
df = pd.DataFrame(frame)
df.to_csv('lda_evaluation.csv')

  