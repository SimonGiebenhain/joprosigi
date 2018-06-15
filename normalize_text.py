import os
import pymorphy2
import re
import pandas as pd

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

morph = pymorphy2.MorphAnalyzer()
retoken = re.compile(r'[\'\w\-]+')
def normalize(text):
    text = retoken.findall(text.lower()) # make all text lowercase
    text = [morph.parse(x)[0].normal_form for x in text] # morphological analysis
    return ' '.join(text)

train['title'] = train['title'].astype(str)
train['description'] = train['description'].astype(str)
test['title'] = test['title'].astype(str)
test['description'] = test['description'].astype(str)

print("Start Normalizing: Train Title")
train['title'] = train['title'].apply(normalize)
print("Start Normalizing: Train Desc")
train['description'] = train['description'].apply(normalize)
print("Start Normalizing: Test Title")
test['title'] = test['title'].apply(normalize)
print("Start Normalizing: Test Desc")
test['description'] = test['description'].apply(normalize)

print('Write out results')
train.to_csv("updnlp-train.csv")
test.to_csv("updnlp-test.csv")