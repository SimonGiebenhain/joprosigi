import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
import category_encoders as ce
import target_encoding as te

color = sns.color_palette()


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

print("Reading Data")
train_df = pd.read_csv("/mnt/0d28870f-7d1c-4a10-841a-0195e11cdce3/Jonas/Avito/input/train.csv", parse_dates=['activation_date'])
test_df  = pd.read_csv("/mnt/0d28870f-7d1c-4a10-841a-0195e11cdce3/Jonas/Avito/test/test.csv", parse_dates=['activation_date'])
print("Finished reading data")
print("Shape train: ", train_df.shape)


# New variable on weekday #
train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

print("Calculating Price:")
# Replace Nan with mean in price
categories = train_df.category_name.unique()
region = train_df.region.unique()
param1 = train_df.param_1.unique()


train_df["price_new"] = train_df["price"].values

for cat in categories:
    #for reg in region:
        cur_df = train_df.loc[(train_df["category_name"] == cat)]["price_new"]  # & (train_df["region"] == reg)]["price_new"]
        cur_df.fillna(np.nanmean(cur_df.values), inplace=True)


train_df["price"] = pd.isna(train_df["price"])
print("Calculated Price")
print(train_df.head())
print("Encoding Labels:")
# Label encode the categorical variables #


train_df['parent_category_name'], test_df['parent_category_name'] = te.target_encode(train_df['parent_category_name'], test_df['parent_category_name'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['category_name'], test_df['category_name'] = te.target_encode(train_df['category_name'], test_df['category_name'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['region'], test_df['region'] = te.target_encode(train_df['region'], test_df['region'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['image_top_1'], test_df['image_top_1'] = te.target_encode(train_df['image_top_1'], test_df['image_top_1'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['city'], test_df['city'] = te.target_encode(train_df['city'], test_df['city'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['param_1'], test_df['param_1'] = te.target_encode(train_df['param_1'], test_df['param_1'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['param_2'], test_df['param_2'] = te.target_encode(train_df['param_2'], test_df['param_2'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['param_3'], test_df['param_3'] = te.target_encode(train_df['param_3'], test_df['param_3'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)
train_df['user_type'], test_df['user_type'] = te.target_encode(train_df['user_type'], test_df['user_type'], train_df['deal_probability'], min_samples_leaf=100, smoothing=10, noise_level=0.01)


train_df.to_csv("/mnt/0d28870f-7d1c-4a10-841a-0195e11cdce3/Jonas/Avito/input/train_encoded.csv")
test_df.to_csv("/mnt/0d28870f-7d1c-4a10-841a-0195e11cdce3/Jonas/Avito/input/test_encoded.csv")

