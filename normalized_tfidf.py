import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import target_encoding as te
import gc

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix, save_npz
from nltk.corpus import stopwords
import time

train_df = pd.read_csv("../data/normalized-text-train.csv", usecols=[ 'description', 'title'])# nrows=1000)
test_df = pd.read_csv("../data/normalized-text-test.csv", usecols=[ 'description', 'title'])#, nrows=1000)

trainindex = train_df.index
testindex = test_df.index

# train_y = train_df.deal_probability.copy()
# train_df.drop("deal_probability",axis=1, inplace=True)

df = pd.concat([train_df,test_df],axis=0)
del train_df, test_df
gc.collect()

textfeats = ["description", "title"]

for cols in textfeats:
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].astype(str).fillna(' ') # FILL NA
    df[cols] = df[cols].str.lower()

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    # "min_df":5,
    # "max_df":.9,
    "smooth_idf": False
}


def get_col(col_name): return lambda x: x[col_name]


vectorizer = FeatureUnion([
    ('description', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=16000,
        **tfidf_para,
        preprocessor=get_col('description'))),
    ('title', TfidfVectorizer(
        ngram_range=(1, 2),
        **tfidf_para,
        # max_features=7000,
        preprocessor=get_col('title')))
])

start_vect = time.time()
vectorizer.fit(df.loc[trainindex, :].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes" % ((time.time() - start_vect) / 60))

# Drop Text Cols
df.drop(textfeats, axis=1, inplace=True)

train_X = hstack([csr_matrix(df.head(trainindex.shape[0]).values),ready_df[0:trainindex.shape[0]]]) # Sparse Matrix
test_X = hstack([csr_matrix(df.tail(testindex.shape[0]).values),ready_df[trainindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [train_X,test_X]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect()

save_npz('../data/tfidf_train.npz', train_X)
save_npz('../data/tfidf_test.npz', test_X)