# 読売新聞のデータセットをプレーンテキスト化
import pandas as pd
import re
import glob
import neologdn
import unicodedata

# 記事のテキスト
texts = ''

# テキストを正規化
def normalized(sent):
  # 正規化
  sent_normalized = neologdn.normalize(sent)
  sent_normalized = unicodedata.normalize('NFKC', sent_normalized)

  # 数字と桁区切り文字を全て0に変換
  sent_normalized = re.sub(r'(\d)([,.])(\d+)', r'\1\3', sent_normalized)
  sent_normalized = re.sub(r'\d+', '0', sent_normalized)

  return sent_normalized

# --------------------------------------------------------
# 探索パスのパターン
text_paths='/home/public/newspapers/yomiuri/*/*.txt'
correct_patterns = 'J[0-9]+utf8.txt'
output_path = '/home/s192c1058/data/yomiuri/'
# 全入力ファイル名
all_input_files = glob.glob(text_paths, recursive=False)
avoid_patterns = ['図＝.{0,25}', '写真＝.{0,25}','<.{0,20}>','＜.{0,20}＞','▽.{0,10}','【.{0,10}】','（ｈｔｔｐ.*）','〈.{0,20}〉']

input_files = []
for infile in all_input_files:
  # 変換済みと修正済みファイルは除外
  if re.search(correct_patterns, infile):
    input_files.append(infile)

# --------------------------------------------------------

# 除外したいワード
stopword=['＼','Ｔ','１','２','／','▲','　','◆','◇','\n']
count=0 # 記事数

# プレーンなテキストに変換
for text_path in input_files:
    with open(text_path, 'r') as f:
        for line in f:
            # 記事（T2）の処理
            if "Ｔ２" in line:
                line = line.rstrip(' ')
                # タグの除去
                for word in stopword:
                    line=line.replace(word,'')
                # 正規表現でパターン除去
                for avoid_pattern in avoid_patterns:
                    if re.search(avoid_pattern, line):
                        line = re.sub(avoid_pattern, '', line)
                
                line_normalized = normalized(line)
                texts+=line_normalized

            # 記事の識別子（T1）の処理
            if "Ｔ１" in line:
                with open(output_path+'newspapers_{}.txt'.format(count),'w')as txtfile:
                    txtfile.write(texts)
                    #print(output_path+'newspapers_{}.txt'.format(count))
                count+=1
                texts='' # テキストを初期化