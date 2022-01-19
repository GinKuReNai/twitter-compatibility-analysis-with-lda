import pandas as pd
import pickle   # モデルを保存
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA # LDA Modeling
from sklearn.feature_extraction.text import TfidfTransformer # BoW / Tfidf
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ----------------------------------------------------------------

#WordCloudを表示
def print_topics(model, count_vectorizer, n_top_words):
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print(topic)
        long_string = ','.join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        font_path='msgothic.ttc'
        wordcloud = WordCloud(font_path=font_path,background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
        wordcloud.generate(long_string)
        wordcloud.to_file(f"pictures/wordcloud_{topic_idx}.png")

# -----------------------------------------------------------------
#LDAのロード
ldaname="models/lda.sav"
loaded_model = pickle.load(open(ldaname, 'rb'))

#tfidfのロード
Tfidfname= 'models/Tfidf.sav'
vectorizer=None
with open(Tfidfname, 'rb') as f:
    vectorizer=pickle.load(f)

#WordCloudの表示
number_words=100
print_topics(loaded_model, vectorizer, number_words) 