# GensimでLDAトピックモデリング
from collections import defaultdict
from gensim import models, corpora
import unicodedata # 正規化用
import neologdn # 正規化用
import glob
import sys
import MeCab
import re

# ------------------------------------------------------

# 探索パスのパターン
path_pattern = '../aozorabunko_text/cards/*/files/*_ruby_*/*_utf-8_revised.txt'
# 入力ファイル名
inputfile = glob.glob(path_pattern, recursive=False)
# MeCabの初期化
mc = MeCab.Tagger("/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
mc.parse('')

# ストップワードリスト + 正規表現パターン
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

  node = mc.parseToNode(sent)
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

# -------------------------------------------------
          
# 形態素解析した結果をtextsに追加
for i in range(len(inputfile)):
  with open(inputfile[i], 'r') as f:
    for line in f:
      texts.append(get_tokens(line))

# ストップワードを削除
texts = del_stopwords(texts, slothlib_stopwords)

# 形態素解析の結果を出力
with open('neologd_output.txt','w') as f:
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
dictionary.filter_extremes(no_below=1, no_above=0.15)
#print(dictionary.token2id)

# BoWを作成
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
#print(bow_corpus)

# Tfidfを作成
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# LDAを作成
N = 10 # トピック数
corpus_lda = lda_model[corpus_tfidf]

# モデルを保存
lda_model.save(f'models/aozorabunko_{N}.model')

# トピックを出力
for i in range(N):
  print('TOPIC: ',i,'_',lda_model.print_topic(i))