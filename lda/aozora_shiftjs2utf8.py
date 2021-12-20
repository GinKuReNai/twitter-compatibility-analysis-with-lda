# 青空文庫のテキストをShift-JIS(CP932)→UTF-8に変換する
import re  # 正規表現
import glob  # ファイルパスでの指定ファイルを正規表現で乱獲

# --------------------------------------------------------
# 探索パスのパターン
path_pattern = '../aozorabunko_text/cards/*/files/*_ruby_*/*.txt'
correct_patterns = '[0-9]+_ruby_[0-9]+.txt'
# 全入力ファイル名
all_input_files = glob.glob(path_pattern, recursive=False)

input_files = []
output_files = []
for infile in all_input_files:
  # 変換済みと修正済みファイルは除外
  if re.search(correct_patterns, infile):
    input_files.append(infile)
    output_files.append(infile.replace('.txt', '_utf-8.txt'))

#----------------------------------------------------------

# エンコード作業
for i in range(len(input_files)):
  fin = open(input_files[i], 'r', encoding='cp932', errors='ignore')
  fout = open(output_files[i], 'w', encoding='utf-8')
  for line in fin:
    fout.write(line)

 