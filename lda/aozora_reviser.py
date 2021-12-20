# 青空文庫のtxtを形態素解析しやすいように1行のプレーンテキストに変換する
import re
import glob

# -------------------------------------------------------
# 探索パスのパターン
path_pattern = '../aozorabunko_text/cards/*/files/*_ruby_*/*.txt'
correct_patterns = '[0-9]+_(ruby|txt)_[0-9]+_utf-8.txt'
# 全入力ファイル名
all_input_files = glob.glob(path_pattern, recursive=False)

input_files = []
output_files = []
for infile in all_input_files:
  # 変換済みと修正済みファイルは除外
  if re.search(correct_patterns, infile):
    input_files.append(infile)
    output_files.append(infile.replace('.txt', '_revised.txt'))

# --------------------------------------------------------

# 正規表現パターン集
end_pattern = '底本：「.+」.+'
patterns = ['-+','【.*】','https://.*','http://.*','《》：.*','｜：.*','［＃］：.*','（例）.*','／＼：.*','〔〕：.*','《.{1,10}》','〔.{1,10}〕','(\*[0-9]+)','\[.*\]','［.{1,40}］','［.*\]','\[\*］','（.*）','※','｜','／＼','\n']

# --------------------------------------------------------
# 変換処理
for i in range(len(input_files)):
  # 修正前ファイル（UTF-8）
  fin = open(input_files[i], 'r', encoding='utf-8', errors='ignore')
  print(input_files[i])
  # 修正後ファイル（UTF-8)
  fout = open(output_files[i], 'w', encoding='utf-8')
  num = 0 # 行数

  for line in fin:
    line = line.rstrip()
    # 最初の2行を除外
    if num < 2:
      num += 1
      continue
    # 最初と最後のパターンではない場合
    if not re.search(end_pattern, line):
      # 邪魔なパターンを削除
      for pattern in patterns:
        if re.search(pattern, line):
          line = re.sub(pattern, '', line)
      # ファイル出力 
      fout.write(line)
    else:
      break