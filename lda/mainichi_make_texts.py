# 毎日新聞のデータセットをプレーンテキスト化
import pandas as pd
import re
import glob

# 記事のテキスト
texts = ''

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

# --------------------------------------------------------

# 除外したいワード
stopword=['＼','Ｔ','２','／','▲','　']
count=0 # 記事数

# プレーンなテキストに変換
for text_path in input_files:
    with open(text_path, 'r') as f:
        for line in f:
            # 記事（T2）の処理
            if "Ｔ２" in line:
                for word in stopword:
                    line=line.replace(word,'')
                texts+=line

            # 記事の識別子（AD）の処理
            if "ＡＤ" in line:
                with open(output_path+'newspapers_{}.txt'.format(count),'w')as txtfile:
                    txtfile.write(texts)
                count+=1
                texts='' # テキストを初期化
            
            
           