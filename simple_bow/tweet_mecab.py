import tweepy # Twitter API
import csv # CSV I/O
import MeCab
import numpy as np
import matplotlib.pyplot as plt # 結果のグラフを描画

# ------------------------------------------

def average_out(genre_count):
  sum = 0
  for i in range(len(genre_count)):
    sum += int(genre_count[i])
  for i in range(len(genre_count)):
    average_count[i] = int(genre_count[i]) / sum * 100
  print('-----------')
  print(average_count)
  print('-----------')

  return average_count

# 円グラフの表示
def roundgraph_show(ndarray):
  label = ["international", "economics", "home", "culture", "reading", "science", "life", "entertainment", "sports", "social"]
  plt.pie(average_ndarray, labels=label, counterclock=False, startangle=90, autopct="%.1f%%")
  plt.show()

# 棒グラフの表示
def bargraph_show(ndarray):
  label = ["international", "economics", "home", "culture", "reading", "science", "life", "entertainment", "sports", "social"]
  left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  plt.bar(left, ndarray, tick_label=label, align="center", color="orange")
  plt.xlabel("Categories")
  plt.ylabel("the number of words")
  plt.grid(True)
  plt.show()

# ---------------------------------------------------

# Twitter API Key
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_SECRET = ""

mc = MeCab.Tagger("-Ochasen")
#mc = MeCab.Tagger("-d ../../../usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
mc.parse('')
dictionary = {}

auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN,ACCESS_SECRET)

api = tweepy.API(auth)

tweet_data = []
words = []
genre_count = [0,0,0,0,0,0,0,0,0,0]
average_count = [0,0,0,0,0,0,0,0,0,0]

# Tweetを取得して形態素解析
for tweet in tweepy.Cursor(api.user_timeline,screen_name = "hirox246",exclude_replies = True).items():
  # tweet_data.append([tweet.id,tweet.created_at,tweet.text.replace('\n',''),tweet.favorite_count,tweet.retweet_count])
  node = mc.parseToNode(tweet.text)
  while node:
    if node.surface: # BOSとEOSを除く
      if ("名詞" or "動詞") in node.feature:
        words.append(node.surface)
    node=node.next
  # tweet_data.append([tweet.text.replace('\n','')])
 
# 毎日新聞（ラベル付き）から取得した情報(mainichi.csv)を解析結果に適用する
with open('mainichi.csv', 'r', encoding='utf-8') as f:
  reader = csv.reader(f) 
  dictionary = {rows[0]:list(rows[1].strip('[]').split(',')) for rows in reader}
  # row[0] : 単語, row[1] : BoW
  
  # 頻度をカウント
  for i in range(len(words)):
    bow = list(dictionary.get(words[i], '0000000000'))
    for i in range(len(genre_count)):
      genre_count[i] += int(bow[i])
  print(genre_count)
  average_count = average_out(genre_count)

  # numpyの配列に変換
  genre_ndarray = np.array(genre_count)
  average_ndarray = np.array(average_count)

  # グラフを表示
  bargraph_show(genre_ndarray)
  roundgraph_show(average_ndarray)
