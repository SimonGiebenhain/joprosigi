import pandas as pd
import numpy as np
from zipfile import ZipFile
import os

print("Reading Data")
data_dir = "../data/"
train_data = pd.read_csv(data_dir+"/train.csv", parse_dates=["activation_date"]) #we will eventually turn the date column into day of week [0,6]
test_data  = pd.read_csv(data_dir+"/test.csv", parse_dates=["activation_date"])
print(test_data.image.head())
#TODO alternatives?
train_data = train_data.replace(np.nan,-1,regex=True) #nan and other missing values are mapped to -1
test_data  = test_data.replace(np.nan,-1,regex=True)
train_t = train_data['title'].fillna('fillna')
test_t = test_data['title'].fillna('fillna')
train_d = train_data['description'].astype(str).fillna('fillna')
test_d = test_data['description'].astype(str).fillna('fillna')

#TODO what about day of month/year
##================Replace Full Dates with Day-of-Week
train_data['week_day'] = train_data["activation_date"].dt.weekday
train_data['month_day'] = train_data["activation_date"].dt.day
test_data['week_day'] = test_data["activation_date"].dt.weekday
test_data['month_day'] = test_data["activation_date"].dt.day

##================Remove unwanted columns
#TODO Discard user_id as well?
del train_data['item_id'],test_data['item_id'], train_data['activation_date'], test_data['activation_date']

#TODO: Or treat image_top_1 as continuos variable?
#TODO item_seqw_num cate or cont?
cat_cats = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3",
            "user_id", "item_seq_number", "user_type", "image_top_1"]

for cat in cat_cats:
    train_data[cat] = pd.Categorical(train_data[cat])
    train_data[cat] = train_data[cat].cat.codes
    test_data[cat] = pd.Categorical(test_data[cat])
    test_data[cat] = test_data[cat].cat.codes

#TODO: What does -inf stand for
train_data.price = train_data.replace(-np.inf, 0)
test_data.price = test_data.replace(-np.inf, 0)

# TODO check if this is working!
min_price = np.min([np.log1p(train_data.price.min(axis=0)), np.log1p(test_data.price.min(axis=0))])
max_price = np.max([np.log1p(train_data.price.max(axis=0)), np.log1p(test_data.price.max(axis=0))])

train_data['price'] = train_data.price.apply(lambda x: (np.log1p(x)-min_price) / (max_price - min_price))
test_data['price'] = test_data.price.apply(lambda x: (np.log1p(x)-min_price) / (max_price - min_price))
print(train_data.nunique())
##================split into x_train/x_val. No stratification requried probably
val_split = 0.1
#shuffle data
train_data = train_data.sample(frac=1).reset_index(drop=True)
val_ix = int(np.rint(len(train_data)*(1.-val_split)))
#data frame formats with y-values packed in
train_df = train_data[:val_ix]
val_df = train_data[val_ix:]
test_df = test_data
print(train_df.shape)

train_df.to_csv('../data/nn_train.csv')
val_df.to_csv('../data/nn_val.csv')
test_df.to_csv('../data/nn_test.csv')