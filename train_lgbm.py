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
from scipy.sparse import hstack, csr_matrix, load_npz
from nltk.corpus import stopwords
import time

train_df = pd.read_csv("../data//train.csv", parse_dates=["activation_date"])#, nrows=1000)
test_df = pd.read_csv("../data/test.csv", parse_dates=["activation_date"])#, nrows=1000)
aggregated_features = pd.read_csv("../data/aggregated_features.csv")
trainindex = train_df.index
testindex = test_df.index
test_id = test_df["item_id"].values
print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)

train_y = train_df.deal_probability.copy()
train_df.drop("deal_probability",axis=1, inplace=True)



# Target encode the categorical variables #
cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3", "image_top_1"]
for col in cat_vars:
    train_df[col], test_df[col] = te.target_encode(train_df[col], test_df[col], train_y, min_samples_leaf=100, smoothing=10, noise_level=0.01)
print('Finished target encoding')
df = pd.concat([train_df, test_df], axis=0)
del train_df, test_df
gc.collect()
df = df.merge(aggregated_features, on=['user_id'], how='left')
del aggregated_features
gc.collect()
# Time Data
df["activation_weekday"] = df["activation_date"].dt.weekday



# Price: replace Nan with category mean
#df["price"] = df.groupby("category_name")['price'].transform(lambda x: x.fillna(x.mean()))
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-1,inplace=True)
df["image_top_1"].fillna(-1,inplace=True)

#Drop Cols
cols_to_drop = ["item_id", "user_id", "activation_date", "image"]
df.drop(cols_to_drop, axis=1,inplace=True)

# Meta Text Features
textfeats = ["description", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].astype(str).fillna(' ') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
df.drop(textfeats, axis=1, inplace=True)

print("Merge with tfidf")
ready_df = load_npz('../data/ready_df.npz')
print(ready_df.shape)
tfvocab = np.load('../data/tfvocab.npy').tolist()

train_X = hstack([csr_matrix(df.head(trainindex.shape[0]).values),ready_df[0:trainindex.shape[0]]]) # Sparse Matrix
test_X = hstack([csr_matrix(df.tail(testindex.shape[0]).values),ready_df[trainindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [train_X,test_X]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect()


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 300,
        "learning_rate": 0.01,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.5,
        "bagging_freq": 2,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y, feature_name=tfvocab)
    lgval = lgb.Dataset(val_X, label=val_y, feature_name=tfvocab)
    evals_result = {}
    model = lgb.train(params, lgtrain, 16000, valid_sets=[lgval], early_stopping_rounds=300, verbose_eval=20,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

# Splitting the data for model training#
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=42)


# Training the model #
pred_test, model, evals_result = run_lgb(X_train, y_train, X_val, y_val, test_X)

# Making a submission file #
pred_test[pred_test>1] = 1
pred_test[pred_test<0] = 0
sub_df = pd.DataFrame({"item_id":test_id})
sub_df["deal_probability"] = pred_test
sub_df.to_csv("normalized_tfidf_aggregated_features.csv", index=False)

fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.savefig("normalized_tfidf_aggregated_features.png")

