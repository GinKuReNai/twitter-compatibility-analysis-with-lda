# 毎日新聞のデータセットをプレーンテキスト化
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
text_paths='/home/public/newspapers/mainichi/*/*.txt'
correct_patterns = 'mai[0-9]+utf8.txt'
output_path = '/home/s192c1058/data/'
# 全入力ファイル名
all_input_files = glob.glob(text_paths, recursive=False)

input_files = []
for infile in all_input_files:
  # 変換済みと修正済みファイルは除外
  if re.search(correct_patterns, infile):
    input_files.append(infile)

# 正規表現
delete_patterns = ['<.{1,30}>','【.{1,30}】','＜.{1,30}＞']
# --------------------------------------------------------

# 除外したいワード
stopword=['＼','Ｔ','２','／','▲','◇','◆','　']
count=0 # 記事数

# プレーンなテキストに変換
for text_path in input_files:
    with open(text_path, 'r') as f:
        for line in f:
            if '【現在著作権交渉中の為、本文は表示できません】' in line:
                continue
            
            # 記事（T2）の処理
            line = line.rstrip() # 改行の処理
            if "Ｔ２" in line:
                # ストップワードの除去
                for word in stopword:
                    line=line.replace(word,'')
                # その他不必要な文字列を削除
                for pattern in delete_patterns:
                    if re.search(pattern, line):
                        line = re.sub(pattern, '', line)
                
                line_normalized = normalized(line) # テキストを正規化
                texts+=line_normalized

            # 記事の識別子（AD）の処理
            if "ＡＤ" in line:
                with open(output_path+'newspapers_{}.txt'.format(count),'w')as txtfile:
                    txtfile.write(texts)
                count+=1
                texts='' # テキストを初期化
            
          