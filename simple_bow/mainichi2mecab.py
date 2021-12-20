import MeCab

mc = MeCab.Tagger("-Ochasen")
mc.parse('') #おまじない

texts=[]
extexts=[]
stop_word=['ＫＢ']

def get_tokens(sent):
        node = mc.parseToNode(sent)
        tokens = []
        while node:
            if node.surface: # BOSとEOSを除く
                if "名詞" in node.feature:
                        tokens.append(node.surface)
                    
            node=node.next

        return tokens

file_path=['../public/newspapers/mainichi/1995/mai1995utf8.txt',
        '../public/newspapers/mainichi/2015/zenkoku/mai2015utf8.txt',
        '/home/public/newspapers/mainichi/2016/zenkoku/mai2016utf8.txt',
        '/home/public/newspapers/mainichi/2019/zenkoku/mai2019utf8.txt']

for file in file_path:
    with open(file)as f:
        for line in f:
            if "ＫＢ" in line:
            # textは助詞を除いた文字列 
                texts.append(get_tokens(line))          

for text in texts:
    for word in text:
        if word not in stop_word:
            extexts.append(word)
            

with open("mainichi_mecab.txt","w",encoding='utf-8')as f:
    f.write(str(extexts))