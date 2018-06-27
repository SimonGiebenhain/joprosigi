import pandas as pd
import numpy as np
import time
import gc
import os
import warnings
import string
import pickle
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import layers, activations
from keras.utils import Sequence
from keras.preprocessing import image
from keras.applications import xception
from zipfile import ZipFile
from gensim.models import word2vec


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from sklearn.model_selection import KFold


from keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
cores = 4

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def add_common_layers(y, name):
    if name:
        y = layers.BatchNormalization(name=name)(y)
    else:
        y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    return y


exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1


dtypes_train = {
                'price': 'float32',
                'item_seq_number': 'uint32',
                'deal_probability': 'float32',
                'image': 'str'
}
dtypes_test = {
                'price': 'float32',
                'item_seq_number': 'uint32',
                'image': 'str'
}
use_cols = ['item_id', 'user_id', 'image_top_1', 'region', 'city', 'parent_category_name',
            'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price',
            'item_seq_number', 'activation_date', 'image', 'user_type', 'deal_probability']

EMBEDDING_DIM1 = 300  # this is from the pretrained vectors
max_num_words = 400000
seq_length = 150

max_region = 28
#max_city = 1752 without region city concat
max_city = 1825
max_category_name = 47
max_parent_category_name = 9
max_param_1 = 372
max_param123 = 2402
max_image_code = 3064
max_day = 32
max_wday = 8
max_user_type = 4



#TODO
#fill price with average of category
#treat image_top_1 as numerical, categorical or string?
#use activation weekday'''

def ztransform(col):
    mean = np.mean(col)
    std = np.std(col)
    return (col - mean) / std


def preprocess_dataset(dataset):
    print("Filling Missing Values.....")

    count = lambda l1, l2: sum([1 for x in l1 if x in l2])

    #Remember index of missing titles and descriptions,
    # such that their words_vs_unique fields can be downgraded from the best possible value
    desc_na_index = dataset['description'].index[dataset['description'].isna()]
    tit_na_index = dataset['title'].index[dataset['title'].isna()]

    dataset['description'].fillna('unknowndescription', inplace=True)
    dataset['title'].fillna('unknowntitle', inplace=True)
    #TODO include weekday! This is day of month or something
    dataset['day'] = pd.to_datetime(dataset['activation_date']).dt.day
    dataset['wday'] = pd.to_datetime(dataset['activation_date']).dt.weekday
    for col in ['description', 'title']:
        dataset['num_words_' + col] = dataset[col].apply(lambda comment: len(comment.split()))
        dataset['num_unique_words_' + col] = dataset[col].apply(lambda comment: len(set(w for w in comment.split())))
    dataset['words_vs_unique_title'] = dataset['num_unique_words_title'] / dataset['num_words_title']
    dataset['words_vs_unique_description'] = dataset['num_unique_words_description'] / dataset['num_words_description']
    dataset['city'] = dataset['region'] + '_' + dataset['city']
    dataset['num_desc_punct'] = dataset['description'].apply(lambda x: count(x, set(string.punctuation)))

    #Downgrade words_vs_unique score for ads missing title or description
    dataset.loc[desc_na_index, 'words_vs_unique_description'] = 0
    dataset.loc[tit_na_index, 'words_vs_unique_title'] = 0

    #Normalize features generated above:
    #TODO: is log1p also worth if there is only small differnce between values?
    #dataset['num_words_title'] = np.log1p(dataset.num_words_title.values)
    dataset['num_words_title'] = np.log1p(dataset.num_words_title.values)
    dataset['num_words_description'] = np.log1p(dataset.num_words_description)
    dataset['num_unique_words_title'] = np.log1p(dataset.num_unique_words_title.values)
    dataset['num_unique_words_description'] = np.log1p(dataset.num_unique_words_description.values)
    dataset['words_vs_unique_title'] = np.log1p(dataset.words_vs_unique_title.values)
    dataset['words_vs_unique_description'] = np.log1p(dataset.words_vs_unique_description.values)
    dataset['num_desc_punct'] = np.log1p(dataset.num_desc_punct.values)


    #dataset['avg_days_up_user'] = dataset['avg_days_up_user'].fillna(0).astype('uint32')
    #dataset['avg_times_up_user'] = dataset['avg_times_up_user'].fillna(0).astype('uint32')
    #dataset['n_user_items'] = dataset['n_user_items'].fillna(0).astype('uint32')

    dataset['param_1'].fillna(value='missing', inplace=True)
    dataset['param_2'].fillna(value='missing', inplace=True)
    dataset['param_3'].fillna(value='missing', inplace=True)

    dataset['param_1'] = dataset['param_1'].astype(str)
    dataset['param_2'] = dataset['param_2'].astype(str)
    dataset['param_3'] = dataset['param_3'].astype(str)

    print("Casting data types to type Category.......")
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['parent_category_name'] = dataset['parent_category_name'].astype('category')
    dataset['region'] = dataset['region'].astype('category')
    dataset['city'] = dataset['city'].astype('category')
    dataset['user_type'] = dataset['user_type'].astype('category')

    dataset['image_top_1'] = dataset['image_top_1'].fillna('missing')
    dataset['image_code'] = dataset['image_top_1'].astype('str')
    del dataset['image_top_1']
    gc.collect()

    #dataset['week'] = pd.to_datetime(dataset['activation_date']).dt.week.astype('uint8')
    #dataset['day'] = pd.to_datetime(dataset['activation_date']).dt.day.astype('uint8')
    #dataset['wday'] = pd.to_datetime(dataset['activation_date']).dt.dayofweek.astype('uint8')
    del dataset['activation_date']
    gc.collect()

    print("Creating New Feature.....")
    dataset['param123'] = (dataset['param_1'] + '_' + dataset['param_2'] + '_' + dataset['param_3']).astype(str)

    dataset['title_description'] = (dataset['param_1'] + ' ' + dataset['param_2'] + ' ' + dataset['param_3'] + ' ' + dataset['title'] + ' ' + dataset['description']).astype(str)
    del dataset['param_2'], dataset['param_3'], dataset['title'], dataset['description']

    gc.collect()

    #Indicator column wheter price was present or not
    dataset['no_price'] = dataset['price'].isna()
    dataset['no_image'] = dataset['image'].isnull()
    #Replace NaNs in price with mean price for that category
    mean_price_df = dataset.groupby('category_name', as_index=False)['price'].mean()
    mean_price_df['mean_price'] = mean_price_df['price']
    print(mean_price_df.head())
    del mean_price_df['price']
    dataset = dataset.merge(mean_price_df, on=['category_name'], how='left')
    dataset['price'] = dataset['price'].fillna(dataset['mean_price'])
    del dataset['mean_price']

    #TODO: NA for aggregated features: indicator plus mean
    dataset['no_avg_days_up_user'] = dataset['avg_days_up_user'].isna()
    dataset['no_avg_times_up_user'] = dataset['avg_times_up_user'].isna()
    dataset['no_n_user_items'] = dataset['n_user_items'].isna()

    mean_user_feat_df = dataset.groupby('user_type', as_index=False)['avg_days_up_user', 'avg_times_up_user', 'n_user_items'].mean()
    mean_user_feat_df['avg_days'] = mean_user_feat_df['avg_days_up_user']
    mean_user_feat_df['avg_times'] = mean_user_feat_df['avg_times_up_user']
    mean_user_feat_df['n_items'] = mean_user_feat_df['n_user_items']
    del mean_user_feat_df['avg_days_up_user'], mean_user_feat_df['avg_times_up_user'],  mean_user_feat_df['n_user_items']
    dataset = dataset.merge(mean_user_feat_df, on='user_type', how='left')
    dataset['avg_days_up_user'].fillna(dataset['avg_days'], inplace=True)
    dataset['avg_times_up_user'].fillna(dataset['avg_times'], inplace=True)
    dataset['n_user_items'].fillna(dataset['n_items'], inplace=True)
    del dataset['avg_days'], dataset['avg_times'], dataset['n_items']
    gc.collect()

    #First scale with log1p, bc. this is more natural for humans, then Z-transform
    dataset['price'] = np.log1p(dataset['price'])
    #dataset['price'] = ztransform(dataset.price.values)

    dataset['avg_days_up_user'] = np.log1p(dataset['avg_days_up_user'])
    #dataset['avg_days_up_user'] = ztransform(dataset.avg_days_up_user.values)

    dataset['avg_times_up_user'] = np.log1p(dataset['avg_times_up_user'])
    #dataset['avg_times_up_user'] = ztransform(dataset.avg_times_up_user.values)

    dataset['n_user_items'] = np.log1p(dataset['n_user_items'])
   # dataset['n_user_items'] = ztransform(dataset.n_user_items.values)

    #bins = pd.IntervalIndex.from_tuples([(-1,2), (2,5), (5,10), (10,20), (20,50), (50,100), (100,200), (200,500), (500,1000), (1000,1500), (1500,10000000)])
    bins = [-1,2,4,7,10,15,20,40,70,100,200,500,700,1000,1500,100000000]
    labels = range(0,15)
    dataset['item_seq_bin'] = pd.cut(dataset['item_seq_number'], bins=bins, labels=labels)
    dataset['item_seq_number'] = np.log1p(dataset['item_seq_number'])
    #dataset['item_seq_number'] = ztransform(dataset.item_seq_number.values)

    dataset['avg_times_up_cat'] = np.log1p(dataset.avg_times_up_cat.values)
    dataset['avg_days_up_cat'] = np.log1p(dataset.avg_days_up_cat.values)
    dataset['n_cat_items'] = ztransform(np.log1p(dataset.n_cat_items.values))

    print("PreProcessing Function completed.")

    return dataset


def set_up_tokenizer(df):
    t = df['title_description'].values

    tokenizer = text.Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(list(t))
    return df, tokenizer

#TODO
#Why labelencoder for param, image_top one, if they were transformed to strings first?
#max_words_title_description
# TODO: convert to lower case first. Or is it default?
def enc_and_normalize(df):
    print("Start Label Encoding process....")
    le_region = LabelEncoder()
    le_region.fit(df.region)

    le_city = LabelEncoder()
    le_city.fit(df.city)

    le_category_name = LabelEncoder()
    le_category_name.fit(df.category_name)

    le_parent_category_name = LabelEncoder()
    le_parent_category_name.fit(df.parent_category_name)

    le_param_1 = LabelEncoder()
    le_param_1.fit(df.param_1)

    le_param123 = LabelEncoder()
    le_param123.fit(df.param123)

    le_image_code = LabelEncoder()
    le_image_code.fit(df.image_code)

    # le_week = LabelEncoder()
    # le_week.fit(DF.week)
    le_day = LabelEncoder()
    le_day.fit(df.day)
    le_wday = LabelEncoder()
    le_wday.fit(df.wday)

    le_user_type = LabelEncoder()
    le_user_type.fit(df.user_type)

    #le_imagenet = LabelEncoder()
    #le_imagenet.fit(pd.concat([df.xception, df.inception, df.inception_resnet], axis=0))


    print("Fit on Train Function completed.")


    df['region'] = le_region.transform(df['region'])
    df['city'] = le_city.transform(df['city'])
    df['category_name'] = le_category_name.transform(df['category_name'])
    df['parent_category_name'] = le_parent_category_name.transform(df['parent_category_name'])
    df['param_1'] = le_param_1.transform(df['param_1'])
    df['param123'] = le_param123.transform(df['param123'])
    df['day'] = le_day.transform(df['day'])
    # dataset['week'] = le_week.transform(dataset['week'])
    df['wday'] = le_wday.transform(df['wday'])
    df['image_code'] = le_image_code.transform(df['image_code'])
    df['user_type'] = le_user_type.transform(df['user_type'])
    #df['xception'] = le_imagenet.transform(df['xception'])
    #df['inception'] = le_imagenet.transform(df['inception'])
    #df['inception_resnet'] = le_imagenet.transform(df['inception_resnet'])



    print("Transform on test function completed.")

    return df



class data_sequence(Sequence):
    def __init__(self, x_set, batch_size, zip_path, tokenizer):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path
        self.tokenizer = tokenizer

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        image_hashs = batch_x.image.values
        if (idx+1) * self.batch_size > len(self.x.index):
            images = np.zeros((len(self.x.index) - idx*self.batch_size, 299, 299, 3), dtype=np.float32)
        else:
            images = np.zeros((self.batch_size, 299, 299, 3), dtype=np.float32)
        with ZipFile(self.zip_path) as im_zip:
            for i,hash in enumerate(image_hashs):
                    try:
                        file = im_zip.open('data/competition_files/train_jpg/' + hash + '.jpg')
                        img = image.load_img(file, target_size=(299,299))
                        img = image.img_to_array(img)
                        if not(np.isinf(img).any() or np.isnan(img).any()):
                            img = xception.preprocess_input(img)
                        images[i] = img
                    except KeyError:
                        print('Error loading image %s' % hash)
                    except IOError:
                        print('Error loading image %s' % hash)
                    except TypeError:
                        pass
        t = self.tokenizer.texts_to_sequences(batch_x['title_description'].astype('str'))

        X = {'input_1': images,
            'seq_title_description': sequence.pad_sequences(t, maxlen=seq_length),
            'region': batch_x.region.values,
            'city': batch_x.city.values,
            'day': batch_x.day.values,
            'wday': batch_x.wday.values,
            'user_type': batch_x.user_type.values,
            'no_price': batch_x.no_price.values,
            'no_image': batch_x.no_image.values,
            'category_name': batch_x.category_name.values,
            'parent_category_name': batch_x.parent_category_name.values,
            'param_1': batch_x.param_1.values,
            'param123': batch_x.param123.values,
            'image_code': batch_x.image_code.values,
            'price': batch_x.price.values,
            'item_seq_number': batch_x.item_seq_number.values,
            'item_seq_bin': batch_x.item_seq_bin.values,
            'avg_ad_days': batch_x.avg_days_up_user.values,
            'no_avg_ad_days': batch_x.no_avg_days_up_user.values,
            'avg_ad_times': batch_x.avg_times_up_user.values,
            'no_avg_ad_times': batch_x.no_avg_times_up_user.values,
            'n_user_items': batch_x.n_user_items.values,
            'no_n_user_items': batch_x.no_n_user_items.values,
            'num_words_title': batch_x.num_words_title.values,
            'num_words_description': batch_x.num_words_description.values,
            'num_unique_words_title': batch_x.num_unique_words_title.values,
            'num_unique_words_description': batch_x.num_unique_words_description.values,
            'words_vs_unique_title': batch_x.words_vs_unique_title.values,
            'words_vs_unique_description': batch_x.words_vs_unique_description.values,
            'num_desc_punct': batch_x.num_desc_punct.values,
            'avg_days_up_cat': batch_x.avg_days_up_cat.values,
            'avg_times_up_cat': batch_x.avg_times_up_cat.values,
            'n_cat_items': batch_x.n_cat_items,
            'matching_classes': batch_x.matching_classes.values,
            'score_mean': batch_x.score_mean.values,
            'score_std': batch_x.score_std.values,
            'avg_light_mean': batch_x.avg_light_mean.values,
            'avg_light_std': batch_x.avg_light_std.values,
            'std_light_mean': batch_x.std_light_mean.values,
            'std_light_std': batch_x.std_light_std.values,
            'size': batch_x['size'].values,
            'awp': batch_x.awp.values,
            'blurriness': batch_x.blurriness.values

        }

        Y = batch_x.deal_probability.values
        return X, Y


    def set_x_set(self, x_set):
        self.x = x_set


class data_sequence_preds(Sequence):
    def __init__(self, x_set, batch_size, zip_path, tokenizer):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path
        self.tokenizer = tokenizer

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        image_hashs = batch_x.image.values
        if (idx+1) * self.batch_size > len(self.x.index):
            images = np.zeros((len(self.x.index) - idx*self.batch_size, 299, 299, 3), dtype=np.float32)
        else:
            images = np.zeros((self.batch_size, 299, 299, 3), dtype=np.float32)
        with ZipFile(self.zip_path) as im_zip:
            for i,hash in enumerate(image_hashs):
                    try:
                        file = im_zip.open('data/competition_files/test_jpg/' + hash + '.jpg')
                        img = image.load_img(file, target_size=(299,299))
                        img = image.img_to_array(img)
                        if not(np.isinf(img).any() or np.isnan(img).any()):
                            img = xception.preprocess_input(img)
                        images[i] = img
                    except KeyError:
                        print('Error loading image %s' % hash)
                    except IOError:
                        print('Error loading image %s' % hash)
                    except TypeError:
                        pass
        t = self.tokenizer.texts_to_sequences(batch_x['title_description'].astype('str'))

        X = {'input_1': images,
            'seq_title_description': sequence.pad_sequences(t, maxlen=seq_length),
            'region': batch_x.region.values,
            'city': batch_x.city.values,
            'day': batch_x.day.values,
            'wday': batch_x.wday.values,
            'user_type': batch_x.user_type.values,
            'no_price': batch_x.no_price.values,
            'no_image': batch_x.no_image.values,
            'category_name': batch_x.category_name.values,
            'parent_category_name': batch_x.parent_category_name.values,
            'param_1': batch_x.param_1.values,
            'param123': batch_x.param123.values,
            'image_code': batch_x.image_code.values,
            'price': batch_x.price.values,
            'item_seq_number': batch_x.item_seq_number.values,
            'item_seq_bin': batch_x.item_seq_bin.values,
            'avg_ad_days': batch_x.avg_days_up_user.values,
            'no_avg_ad_days': batch_x.no_avg_days_up_user.values,
            'avg_ad_times': batch_x.avg_times_up_user.values,
            'no_avg_ad_times': batch_x.no_avg_times_up_user.values,
            'n_user_items': batch_x.n_user_items.values,
            'no_n_user_items': batch_x.no_n_user_items.values,
            'num_words_title': batch_x.num_words_title.values,
            'num_words_description': batch_x.num_words_description.values,
            'num_unique_words_title': batch_x.num_unique_words_title.values,
            'num_unique_words_description': batch_x.num_unique_words_description.values,
            'words_vs_unique_title': batch_x.words_vs_unique_title.values,
            'words_vs_unique_description': batch_x.words_vs_unique_description.values,
            'num_desc_punct': batch_x.num_desc_punct.values,
            'avg_days_up_cat': batch_x.avg_days_up_cat.values,
            'avg_times_up_cat': batch_x.avg_times_up_cat.values,
            'n_cat_items': batch_x.n_cat_items,
            'matching_classes': batch_x.matching_classes.values,
            'score_mean': batch_x.score_mean.values,
            'score_std': batch_x.score_std.values,
            'avg_light_mean': batch_x.avg_light_mean.values,
            'avg_light_std': batch_x.avg_light_std.values,
            'std_light_mean': batch_x.std_light_mean.values,
            'std_light_std': batch_x.std_light_std.values,
            'size': batch_x['size'].values,
            'awp': batch_x.awp.values,
            'blurriness': batch_x.blurriness.values

        }

        return X


    def set_x_set(self, x_set):
        self.x = x_set


class data_sequence_feat(Sequence):
    def __init__(self, x_set, batch_size, zip_path, tokenizer):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path
        self.tokenizer = tokenizer

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        image_hashs = batch_x.image.values
        if (idx+1) * self.batch_size > len(self.x.index):
            images = np.zeros((len(self.x.index) - idx*self.batch_size, 299, 299, 3), dtype=np.float32)
        else:
            images = np.zeros((self.batch_size, 299, 299, 3), dtype=np.float32)
        with ZipFile(self.zip_path) as im_zip:
            for i,hash in enumerate(image_hashs):
                    try:
                        file = im_zip.open('data/competition_files/train_jpg/' + hash + '.jpg')
                        img = image.load_img(file, target_size=(299,299))
                        img = image.img_to_array(img)
                        if not(np.isinf(img).any() or np.isnan(img).any()):
                            img = xception.preprocess_input(img)
                        images[i] = img
                    except KeyError:
                        print('Error loading image %s' % hash)
                    except IOError:
                        print('Error loading image %s' % hash)
                    except TypeError:
                        pass
        t = self.tokenizer.texts_to_sequences(batch_x['title_description'].astype('str'))

        X = {'input_1': images,
            'seq_title_description': sequence.pad_sequences(t, maxlen=seq_length),
            }

        return X


    def set_x_set(self, x_set):
        self.x = x_set


def add_image_features(train, test):
    def get_matching_classes(row):
        c1 = 'xception'
        c2 = 'inception'
        c3 = 'inception_resnet'
        if row[c1] == row[c2]:
            if row[c1] == row[c3]:
                if row[c1] == 'nematode':
                    return np.NaN
                else:
                    return 2
            else:
                return 1
        elif row[c2] == row[c3]:
            return 1
        else:
            return 0

    def ztransform(col):
        std = col.std()
        if std == 0:
            return col
        else:
            mean = col.mean()
            return col.apply(lambda x: (x - mean) / std)

    def extract_features(base):
        base['matching_classes'] = base[['xception', 'inception', 'inception_resnet']].apply(get_matching_classes,
                                                                                             axis=1)
        cols = ['matching_classes','size', 'awp', 'blurriness', 'xception_score', 'inception_score', 'inception_resnet_score',
                'avg_red', 'avg_green', 'avg_blue', 'std_red', 'std_green', 'std_blue', ]
        df = base.groupby('category_name', as_index=False)[cols].mean()
        for i in range(len(cols)):
            df[cols[i] + '_'] = df[cols[i]]
            del df[cols[i]]
        gc.collect()
        base = base.merge(df, on='category_name', how='left')
        for i in range(len(cols)):
            base[cols[i]].fillna(base[cols[i] + '_'], inplace=True)
            del base[cols[i] + '_']
        gc.collect()
        base['score_mean'] = base[['xception_score', 'inception_score', 'inception_resnet_score']].mean(axis=1)
        base['score_std'] = base[['xception_score', 'inception_score', 'inception_resnet_score']].std(axis=1)
        base['avg_light_mean'] = base[['avg_red', 'avg_green', 'avg_blue']].mean(axis=1)
        base['avg_light_std'] = base[['avg_red', 'avg_green', 'avg_blue']].std(axis=1)
        base['std_light_mean'] = base[['std_red', 'std_green', 'std_blue']].mean(axis=1)
        base['std_light_std'] = base[['std_red', 'std_green', 'std_blue']].std(axis=1)
        for col in cols[4:]:
            del base[col]
        del base['xception'], base['inception'], base['inception_resnet']
        gc.collect()
        cols = cols[:4]
        cols = cols + ['score_mean', 'score_std', 'avg_light_mean', 'avg_light_std', 'std_light_mean', 'std_light_std']
        for col in cols:
            base[col] = ztransform(base[col])
        return base

    im_0 = pd.read_csv('../data/image_features_0_test.csv')
    im_1 = pd.read_csv('../data/image_features_1_test.csv')
    im_2 = pd.read_csv('../data/image_features_2_test.csv')
    im_3 = pd.read_csv('../data/image_features_3_test.csv')
    im_4 = pd.read_csv('../data/image_features_4_test.csv')
    image_features_test = pd.concat([im_0, im_1, im_2, im_3, im_4], axis=0, ignore_index=True)
    del im_0, im_1, im_2, im_3, im_4
    gc.collect()

    test = pd.concat([test, image_features_test], axis=1)

    xception_score_nematode = test.xception_score.values[3]
    inception_score_nematode = test.inception_score.values[3]
    inception_resnet_score_nematode = test.inception_resnet_score.values[3]

    # TODO: Fill -1 properly

    test.replace([-1, xception_score_nematode, inception_score_nematode, inception_resnet_score_nematode], np.NaN,
                 inplace=True)

    test = extract_features(test)
    print(test.isnull().sum().sort_values(ascending=False))

    gc.collect()

    print(train.isnull().sum().sort_values(ascending=False))
    zip_path = '../data/train_jpg.zip'
    files_in_zip = ZipFile(zip_path).namelist()
    item_ids = list(map(lambda s: s.split('/')[-1].split('.')[0], files_in_zip))
    train_with_image = train.loc[train['image'].isin(item_ids)].reset_index(drop=True)

    i_0 = pd.read_csv('../data/image_features_0.csv')
    i_1 = pd.read_csv('../data/image_features_1.csv')
    i_2 = pd.read_csv('../data/image_features_2.csv')
    i_3 = pd.read_csv('../data/image_features_3.csv')
    i_4 = pd.read_csv('../data/image_features_4.csv')
    i_5 = pd.read_csv('../data/image_features_5.csv')
    i_6 = pd.read_csv('../data/image_features_6.csv')
    i_7 = pd.read_csv('../data/image_features_7.csv')
    i_8 = pd.read_csv('../data/image_features_8.csv')
    i_9 = pd.read_csv('../data/image_features_9.csv')

    image_feature_train = pd.concat([i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9], axis=0,
                                    ignore_index=True).reset_index(drop=True)

    train_with_image = pd.concat([train_with_image, image_feature_train], axis=1)

    no_image_index = np.logical_not(train['image'].isin(item_ids))
    train_without_image = train.loc[no_image_index]
    train = pd.concat([train_with_image, train_without_image], axis=0, ignore_index=True)
    train = extract_features(train)
    train = train.sample(frac=1).reset_index(drop=True)

    return train, test


def prep_df():
    train = pd.read_csv("../data/train.csv", parse_dates=["activation_date"], usecols = use_cols, dtype = dtypes_train)
    test = pd.read_csv("../data/test.csv", parse_dates=["activation_date"], usecols=use_cols[:-1], dtype=dtypes_test)

    print(train.isnull().sum().sort_values(ascending=False))
    print(test.isnull().sum().sort_values(ascending=False))

    train, test = add_image_features(train, test)

    print(train.isnull().sum().sort_values(ascending=False))
    print(test.isnull().sum().sort_values(ascending=False))
    print(list(train))
    print(list(test))

    y_train = np.array(train['deal_probability'])
    del train['deal_probability']
    gc.collect()

    train_size = train.shape[0]
    all = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()

    user_features = pd.read_csv('../data/aggregated_features.csv')
    all = all.merge(user_features, on=['user_id'], how='left')
    del user_features
    #del all['item_id'], all['user_id']
    gc.collect()

    cat_features = pd.read_csv('../data/aggregated_cat_feat.csv')
    all = all.merge(cat_features, on=['category_name'], how='left')
    del cat_features
    gc.collect()

    all = preprocess_dataset(all)
    all, tokenizer = set_up_tokenizer(all)
    all = enc_and_normalize(all)
    print("Tokenization done and TRAIN READY FOR Validation splitting")

    # Calculation of max values for Categorical fields
    #TODO hard code, since they are fixed
    #max_region = np.max(all.region.max())+1
    #max_city= np.max(all.city.max())+1
    #max_category_name = np.max(all.category_name.max())+1
    #max_parent_category_name = np.max(all.parent_category_name.max())+1
    #max_param_1 = np.max(all.param_1.max())+1
    #max_param123 = np.max(all.param123.max())+1
    #max_week = np.max(all.week.max())+1
    #max_day = np.max(all.day.max())+1
    #max_wday = np.max(all.wday.max())+1
    #max_image_code = np.max(all.image_code.max())+1
    #print(max_region, max_city, max_category_name, max_parent_category_name, max_param_1, max_param123, max_image_code)

    with open('./others/own_tokenizer.pickle', 'wb') as handle:
       pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO
    # vocabsize + 2
    # dafuq is qoing on with c,c1,w_Y, w_No: WHY?

    EMBEDDING_FILE1 = '../data/embeddings/avito.w2v'

    emb_model = word2vec.Word2Vec.load(EMBEDDING_FILE1)
    word_index = tokenizer.word_index
    nb_words = min(max_num_words,len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM1))

    for word, i in word_index.items():
        if i >= max_num_words: continue
        try:
            embedding_vector = emb_model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    print(" FAST TEXT DONE")
    np.save('./others/own_embedding_matrix_2.npy', embedding_matrix)
    del embedding_matrix
    gc.collect()

    train = all[:train_size]
    train.to_csv('../data/all_nn_train.csv')
    del train
    gc.collect()
    test = all[train_size:]
    test.to_csv('../data/all_nn_test.csv')
    del test
    del all
    gc.collect()
    np.save('../data/all_nn_y_train.npy', y_train)


def RNN_model(get_feat):
    base_model = xception.Xception(weights='imagenet', include_top=False)
    # for layer in base_model.layers[:125]:
    #    layer.trainable = False
    # for layer in base_model.layers[125:]:
    #    layer.trainable = True
    x_base = base_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x_base)
    x = layers.Dense(512)(layers.Dropout(0.5)(x))
    x = add_common_layers(x,'')
    x = layers.Dense(64)(layers.Dropout(0.5)(x))
    x = add_common_layers(x,'')

    # Inputs
    seq_title_description = Input(shape=[seq_length], name="seq_title_description")
    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    day = Input(shape=[1], name="day")
    wday = Input(shape=[1], name="wday")
    user_type = Input(shape=[1], name="user_type")
    no_price = Input(shape=[1], name="no_price")
    no_image = Input(shape=[1], name="no_image")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")
    param_1 = Input(shape=[1], name="param_1")
    param123 = Input(shape=[1], name="param123")
    image_code = Input(shape=[1], name="image_code")
    price = Input(shape=[1], name="price")
    item_seq_number = Input(shape=[1], name='item_seq_number')
    item_seq_bin = Input(shape=[1], name='item_seq_bin')
    avg_ad_days = Input(shape=[1], name="avg_ad_days")
    no_avg_ad_days = Input(shape=[1], name="no_avg_ad_days")
    avg_ad_times = Input(shape=[1], name="avg_ad_times")
    no_avg_ad_times = Input(shape=[1], name="no_avg_ad_times")
    n_user_items = Input(shape=[1], name="n_user_items")
    no_n_user_items = Input(shape=[1], name="no_n_user_items")
    num_words_title = Input(shape=[1], name="num_words_title")
    num_words_description = Input(shape=[1], name="num_words_description")
    num_unique_words_title = Input(shape=[1], name="num_unique_words_title")
    num_unique_words_description = Input(shape=[1], name="num_unique_words_description")
    words_vs_unique_title = Input(shape=[1], name="words_vs_unique_title")
    words_vs_unique_description = Input(shape=[1], name="words_vs_unique_description")
    num_desc_punct = Input(shape=[1], name="num_desc_punct")
    avg_cat_days = Input(shape=[1], name="avg_days_up_cat")
    avg_cat_times = Input(shape=[1], name="avg_times_up_cat")
    n_cat_items = Input(shape=[1], name="n_cat_items")

    matching_classe = Input(shape=[1], name="matching_classes")
    score_mean = Input(shape=[1], name="score_mean")
    score_std = Input(shape=[1], name="score_std")
    avg_light_mean = Input(shape=[1], name="avg_light_mean")
    avg_light_std = Input(shape=[1], name="avg_light_std")
    std_light_mean = Input(shape=[1], name="std_light_mean")
    std_light_std = Input(shape=[1], name="std_light_std")
    size = Input(shape=[1], name="size")
    awp = Input(shape=[1], name="awp")
    blurriness = Input(shape=[1], name="blurriness")


    emb_size_small = 6
    emb_size_medium = 14
    emb_size_big = 50

    # Embeddings layers
    embedding_matrix1 = np.load('./others/own_embedding_matrix_2.npy')
    vocab_size = embedding_matrix1.shape[0]
    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights=[embedding_matrix1],
                                          input_length=seq_length, trainable=False)(seq_title_description)
    #emb_seq_description = Embedding(vocab_size, EMBEDDING_DIM1, weights=[embedding_matrix1],
    #                                      input_length=seq_length_description, trainable=False)(seq_description)

    emb_region = Embedding(max_region, emb_size_medium, name='emb_region')(region)
    emb_city = Embedding(max_city, emb_size_big, name='emb_city')(city)
    emb_day = Embedding(max_day, emb_size_small, name='emb_day')(day)
    emb_wday = Embedding(max_wday, emb_size_small, name='emb_wday')(wday)
    emb_user_type = Embedding(max_user_type, 2, name='emb_user_type')(user_type)
    emb_category_name = Embedding(max_category_name, 20, name='emb_category')(category_name)
    emb_parent_category_name = Embedding(max_parent_category_name, emb_size_small, name='emb_parent_category')(parent_category_name)
    emb_param_1 = Embedding(max_param_1, emb_size_big, name='emb_param1')(param_1)
    emb_param123 = Embedding(max_param123, emb_size_big, name='emb_param123')(param123)
    emb_image_code = Embedding(max_image_code, emb_size_big, name='emb_image_code')(image_code)
    emb_item_seq_bin = Embedding(12, emb_size_small, name='emb_item_seq_bin')(item_seq_bin)

    d = SpatialDropout1D(0.2)(emb_seq_title_description)
    # CuDNNRNN or GRU with recurrent dropout?
    d = Bidirectional(layers.CuDNNGRU(50, return_sequences=True, name='nlp_gru_d'))(d)
    # x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    # x = Dropout(0.2)(x)
    d = SpatialDropout1D(0.15)(d)
    d = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform", name='nlp_conv_d')(d)
    d = add_common_layers(d, 'nlp_bn_d')
    avg_pool_d = GlobalAveragePooling1D()(d)
    max_pool_d = GlobalMaxPooling1D()(d)
    avg_pool_d = Dropout(0.5)(avg_pool_d)
    max_pool_d = Dropout(0.5)(max_pool_d)
    d = concatenate([avg_pool_d, max_pool_d])
    cat_net = concatenate([
        emb_region,
        emb_city,
        emb_day,
        emb_wday,
        emb_user_type,
        emb_category_name,
        emb_parent_category_name,
        emb_param_1,
        emb_param123,
        emb_image_code,
        emb_item_seq_bin
    ])
    cat_net = Flatten()(SpatialDropout1D(0.2)(cat_net))
    # main layer
    cat_net = concatenate([
        cat_net,
        no_price,
        no_image,
        no_avg_ad_days,
        no_avg_ad_times,
        no_n_user_items,
    ])
    #TODO: Add batch norm before activation?
    cat_net = Dense(256, name='cat_dense_0')(cat_net)
    cat_net = Dropout(0.5)(add_common_layers(cat_net, 'cat_bn_0'))
    cat_net = Dense(128, name='cat_dense_1')(cat_net)
    cat_net = Dropout(0.5)(add_common_layers(cat_net, 'cat_bn_1'))
    cat_net = Dense(32, name='cat_dense_2')(cat_net)
    cat_net = Dropout(0.5)(add_common_layers(cat_net, 'cat_bn_2'))
    cont_net = concatenate([cat_net,
                            avg_ad_days,
                            avg_ad_times,
                            n_user_items,
                            price,
                            item_seq_number,
                            num_words_title,
                            num_words_description,
                            num_unique_words_title,
                            num_unique_words_description,
                            words_vs_unique_title,
                            words_vs_unique_description,
                            num_desc_punct,
                            avg_cat_days,
                            avg_cat_times,
                            n_cat_items,
                            matching_classe,
                            score_mean, score_std,
                            avg_light_mean, avg_light_std, std_light_mean, std_light_std,
                            size, awp, blurriness
                            ])
    cont_net = Dense(128, name='cat_dense_3')(cont_net)
    cont_net = Dropout(0.5)(add_common_layers(cont_net, 'cat_bn_3'))
    cont_net = Dense(64, name='cat_dense_4')(cont_net)
    cont_net = Dropout(0.5)(add_common_layers(cont_net, 'cat_bn_4'))
    cont_net = Dense(32, name='cat_dense_5')(cont_net)
    cont_net = Dropout(0.2)(add_common_layers(cont_net, 'cat_bn_5'))
    main_net = concatenate([d, cat_net, cont_net, x])
    main_net = Dense(256, name='no_dense_0')(main_net)
    main_net = Dropout(0.5)(add_common_layers(main_net, 'no_bn_0'))
    main_net = Dense(128, name='no_dense_1')(main_net)
    main_net = Dropout(0.4)(add_common_layers(main_net, 'no_bn_1'))
    main_net = Dense(64, name='no_dense_2')(main_net)
    main_net = Dropout(0.3)(add_common_layers(main_net, 'no_bn_2'))

    # output
    output = Dense(1, activation="sigmoid", name='no_dense_3')(main_net)

    # model
    if not get_feat:
        model = Model([base_model.input, seq_title_description,region, city, day, wday, user_type, no_price, no_image, category_name, parent_category_name,
                       param_1, param123, image_code, price, item_seq_number, item_seq_bin,
                       avg_ad_days, no_avg_ad_days, avg_ad_times, no_avg_ad_times, n_user_items, no_n_user_items,
                       num_words_title,num_words_description, num_unique_words_title, num_unique_words_description,
                       words_vs_unique_title, words_vs_unique_description, num_desc_punct,
                       avg_cat_days, avg_cat_times, n_cat_items,
                       matching_classe,
                       score_mean, score_std,
                       avg_light_mean, avg_light_std, std_light_mean, std_light_std,
                       size, awp, blurriness
                       ],
                      output)
        model.compile(optimizer='adam',
                      loss=root_mean_squared_error)
        model.summary()
    else:
        model = Model([base_model.input, seq_title_description, region, city, day, wday, user_type, no_price, no_image,
                       category_name, parent_category_name,
                       param_1, param123, image_code, price, item_seq_number, item_seq_bin,
                       avg_ad_days, no_avg_ad_days, avg_ad_times, no_avg_ad_times, n_user_items, no_n_user_items,
                       num_words_title, num_words_description, num_unique_words_title, num_unique_words_description,
                       words_vs_unique_title, words_vs_unique_description, num_desc_punct,
                       avg_cat_days, avg_cat_times, n_cat_items,
                       matching_classe,
                       score_mean, score_std,
                       avg_light_mean, avg_light_std, std_light_mean, std_light_std,
                       size, awp, blurriness
                       ],
                      [output,d,x])
        model.compile(optimizer='adam',
                      loss=root_mean_squared_error)
        model.summary()
    return model


def RNN_model_feat():
    base_model = xception.Xception(weights='imagenet', include_top=False)
    # for layer in base_model.layers[:125]:
    #    layer.trainable = False
    # for layer in base_model.layers[125:]:
    #    layer.trainable = True
    x_base = base_model.output
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x_base)
    x = layers.Dense(512)(layers.Dropout(0.5)(x))
    x = add_common_layers(x,'')
    x = layers.Dense(64)(layers.Dropout(0.5)(x))
    x = add_common_layers(x,'')

    # Inputs
    seq_title_description = Input(shape=[seq_length], name="seq_title_description")



    # Embeddings layers
    embedding_matrix1 = np.load('./others/own_embedding_matrix_2.npy')
    vocab_size = embedding_matrix1.shape[0]
    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights=[embedding_matrix1],
                                          input_length=seq_length, trainable=False)(seq_title_description)
    #emb_seq_description = Embedding(vocab_size, EMBEDDING_DIM1, weights=[embedding_matrix1],
    #                                      input_length=seq_length_description, trainable=False)(seq_description)


    d = SpatialDropout1D(0.2)(emb_seq_title_description)
    # CuDNNRNN or GRU with recurrent dropout?
    d = Bidirectional(layers.CuDNNGRU(50, return_sequences=True, name='nlp_gru_d'))(d)
    # x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    # x = Dropout(0.2)(x)
    d = SpatialDropout1D(0.15)(d)
    d = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform", name='nlp_conv_d')(d)
    d = add_common_layers(d, 'nlp_bn_d')
    avg_pool_d = GlobalAveragePooling1D()(d)
    max_pool_d = GlobalMaxPooling1D()(d)
    avg_pool_d = Dropout(0.5)(avg_pool_d)
    max_pool_d = Dropout(0.5)(max_pool_d)
    d = concatenate([avg_pool_d, max_pool_d])

    # model

    model = Model([base_model.input, seq_title_description],
                  [d,x])
    model.compile(optimizer='adam',
                  loss=root_mean_squared_error)
    model.summary()

    return model



def rmse(y, y_pred):
    Rsum = np.sum((y - y_pred) ** 2)
    n = y.shape[0]
    RMSE = np.sqrt(Rsum / n)
    return RMSE


def eval_model(model, X, Y):
    val_preds = model.predict(X)
    y_pred = val_preds[:, 0]

    y_true = np.array(Y)

    yt = pd.DataFrame(y_true)
    yp = pd.DataFrame(y_pred)

    print(yt.isnull().any())
    print(yp.isnull().any())

    v_rmse = rmse(y_true, y_pred)
    print(" RMSE for VALIDATION SET: " + str(v_rmse))
    return v_rmse


def predictions(model):
    test = pd.read_csv('../data/rnn_test.csv')

    with open('./others/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    X_test = get_keras_data(test, tokenizer)
    del test
    gc.collect()

    batch_size = 256 * 3
    preds = model.predict(X_test, batch_size=batch_size, verbose=1)
    del X_test
    gc.collect()
    print("RNN Prediction is done")

    preds = preds.reshape(-1, 1)
    preds = np.clip(preds, 0, 1)
    print(preds.shape)

    return preds




# TODO: what do some variables do?
# TODO why increase batch size?
# TODO why train in loop and not for multiple epochs?
# TODO is additional exponential decay benefitial?
# TODO small or big epochs? bc of callbacks?
# TODO is it a big drawback to only train on a third of the data?
# TODO bigger ensemble
def fold():
    train = pd.read_csv('../data/rnn_train.csv')
    y_train = np.load('../data/rnn_y_train.npy')

    with open('./others/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


    skf = KFold(n_splits=3)
    Kfold_preds_final = []
    k = 0
    RMSE = []

    for train_idx, test_idx in skf.split(train, y_train):

        print("Number of Folds.." + str(k + 1))

        # Initialize a new Model for Current FOLD
        epochs = 10
        batch_size = 512 * 3
        #steps = (int(train.shape[0] / batch_size)) * epochs
        #lr_init, lr_fin = 0.009, 0.002
        #lr_decay = exp_decay(lr_init, lr_fin, steps)
        modelRNN = RNN_model()
        K.set_value(modelRNN.optimizer.lr, 0.002)
        #K.set_value(modelRNN.optimizer.decay, lr_decay)

        # K Fold Split

        X_train, X_val = train.iloc[train_idx], train.iloc[test_idx]
        print(X_train.shape, X_val.shape)
        Y_train, Y_val = y_train[train_idx], y_train[test_idx]
        print(Y_train.shape, Y_val.shape)
        gc.collect()

        X_train = get_keras_data(X_train, tokenizer)
        X_val = get_keras_data(X_val, tokenizer)

        earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=1, verbose=1)
        checkpt = ModelCheckpoint(monitor="val_loss", mode="auto", filepath='weights/rnn_model_baseline_weights.hdf5', verbose=1,
                                      save_best_only=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=0, verbose=1, factor=0.3, cooldown=0, min_lr=1e-6)

        # Fit the NN Model
        modelRNN.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=10,
                    validation_data=(X_val, Y_val),
                    callbacks=[earlystop, checkpt, rlrop],
                    shuffle=True,
                    verbose=1 )

        del X_train
        gc.collect()

        # Print RMSE for Validation set for Kth Fold
        v_rmse = eval_model(modelRNN, X_val, Y_val)
        RMSE.append(v_rmse)

        del X_val
        del Y_train, Y_val
        gc.collect()

        # Predict test set for Kth Fold
        preds = predictions(modelRNN)
        modelRNN.save_weights('./weights/rnn_kfold%d.hdf5' % k)
        del modelRNN
        gc.collect()

        print("Predictions done for Fold " + str(k))
        print(preds.shape)
        Kfold_preds_final.append(preds)
        del preds
        gc.collect()
        print("Number of folds completed...." + str(len(Kfold_preds_final)))
        print(Kfold_preds_final[k][0:10])
        k = k + 1

    print("All Folds completed")
    print("RNN FOLD MODEL Done")

    pred_final1 = np.average(Kfold_preds_final, axis=0) # Average of all K Folds
    print(pred_final1.shape)

    min_value = min(RMSE)
    RMSE_idx = RMSE.index(min_value)
    print(RMSE_idx)
    pred_final2 = Kfold_preds_final[RMSE_idx]
    print(pred_final2.shape)

    #del Kfold_preds_final, train1
    gc.collect()

    test_cols = ['item_id']
    test = pd.read_csv('../data/test.csv', usecols = test_cols)

    # using Average of KFOLD preds

    submission1 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

    submission1['item_id'] = test['item_id']
    submission1['deal_probability'] = pred_final1

    print("Check Submission NOW!!!!!!!!@")
    submission1.to_csv("Avito_Shanth_RNN_AVERAGE.csv", index=False)

    # Using KFOLD preds with Minimum value
    submission2 = pd.DataFrame( columns = ['item_id', 'deal_probability'])

    submission2['item_id'] = test['item_id']
    submission2['deal_probability'] = pred_final2

    print("Check Submission NOW!!!!!!!!@")
    submission2.to_csv("Avito_Shanth_RNN_MIN.csv", index=False)


#TODO: Check whether numerical data is properly normalized!
#Check wheterher order of train and y_train corresponds!
def train_only():
    train = pd.read_csv('../data/all_nn_train.csv')
    print(train.info())
    y_train = np.load('../data/all_nn_y_train.npy')
    train['deal_probability'] = y_train

    zip_path = '../data/train_jpg.zip'
    #files_in_zip = ZipFile(zip_path).namelist()
    #item_ids = list(map(lambda s: s.split('/')[-1].split('.')[0], files_in_zip))
    #train = train.loc[train['image'].isin(item_ids)]

    #TODO: wenn final training train = train.sample(frac=1).reset_index(drop=True)
    train_len = int(len(train.index) * 0.33)
    val = train[-train_len:]
    train = train[:-train_len]
    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)


    with open('./others/own_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Initialize a new Model for Current FOLD
    batch_size = 48
    #steps = (int(train.shape[0] / batch_size)) * epochs
    #lr_init, lr_fin = 0.009, 0.00045
    #lr_decay = exp_decay(lr_init, lr_fin, steps)
    model = RNN_model(False)
    #K.set_value(modelRNN.optimizer.decay, lr_decay)

    #earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=3, verbose=1)
    #checkpt = ModelCheckpoint(monitor="val_loss", mode="auto", filepath='weights/rnn_model_baseline_weights.hdf5',
    #                          verbose=1,
    #                          save_best_only=True)
    #rlrop = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=1, verbose=1, factor=0.1, cooldown=0,
    #                         min_lr=1e-6)
    # Fit the NN Model
    #TODO: does this work?
    model.load_weights('./weights/image_nn_serve_27.hdf5',by_name=True)#, skip_mismatch=True)
    #TODO especially are wrong layers loaded? does it matter?

    model.load_weights('./weights/rnn_greedy.hdf5', by_name=True)#, skip_mismatch=True)
    #gru_weights = np.load('./weights/gru.npy')
    #model.get_layer('GRU').set_weights(gru_weights)

    #TODO
    nb_to_train_dense = range(0,4)
    to_train_dense = list(map(lambda x: 'no_dense_%d' % x, nb_to_train_dense))
    nb_to_train_batch_normalization = range(0,3)
    to_train_batch_normalization = list(map(lambda x: 'no_bn_%d' % x, nb_to_train_batch_normalization))

    for layer in model.layers:
        layer.trainable = False
    for name in to_train_dense:
        layer = model.get_layer(name=name)
        layer.trainable = True
    for name in to_train_batch_normalization:
        layer = model.get_layer(name=name)
        layer.trainable = True

    #model.get_layer('load_dense_3').trainable = True

    lr = 0.005
    #TODO: learning rate?
    model.compile(optimizer=optimizers.Adam(lr=0.005),
                  loss=root_mean_squared_error)
    model.summary()

    data_seq_train = data_sequence(train, batch_size, zip_path, tokenizer)
    data_seq_val = data_sequence(val, batch_size, zip_path, tokenizer)

    past_best = 0
    # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
    it = 50
    pat = 0
    early_stop = 0


    for i in range(it):
        print('Iteration %d begins' % i)
        model.fit_generator(data_seq_train, steps_per_epoch=200, epochs=1,
                            verbose=0,
                            max_queue_size=10, workers=8, use_multiprocessing=True, )

        hist = model.evaluate_generator(data_seq_val, steps=18,
                                        max_queue_size=12, workers=6, use_multiprocessing=True)
        train = train.sample(frac=1).reset_index(drop=True)
        val = val.sample(frac=1).reset_index(drop=True)
        data_seq_train.set_x_set(train)
        data_seq_val.set_x_set(val)
        if i > 0:
            curr_loss = hist
            if curr_loss < past_best:
                model.save_weights('./weights/all_nn_good.hdf5')
                print('Loss improved from %f to %f!' % (past_best, curr_loss))
                past_best = hist
            else:
                pat += 1
                print('Loss did not improve from %f!' % past_best)
                if pat == 3:
                    lr /= 5
                    K.set_value(model.optimizer.lr, lr)
                    pat = 0
                    early_stop += 1
                    if early_stop == 3:
                        break

        else:
            past_best = hist
        print(hist)
        gc.collect()


def continue_train():
    train = pd.read_csv('../data/all_nn_train.csv')
    print(train.info())
    y_train = np.load('../data/all_nn_y_train.npy')
    train['deal_probability'] = y_train

    zip_path = '../data/train_jpg.zip'
    #files_in_zip = ZipFile(zip_path).namelist()
    #item_ids = list(map(lambda s: s.split('/')[-1].split('.')[0], files_in_zip))
    #train = train.loc[train['image'].isin(item_ids)]

    #TODO: wenn final training train = train.sample(frac=1).reset_index(drop=True)
    train_len = int(len(train.index) * 0.1)
    val = train[-train_len:]
    train = train[:-train_len]
    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)


    with open('./others/own_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Initialize a new Model for Current FOLD
    batch_size = 32
    #steps = (int(train.shape[0] / batch_size)) * epochs
    #lr_init, lr_fin = 0.009, 0.00045
    #lr_decay = exp_decay(lr_init, lr_fin, steps)
    model = RNN_model(False)
    #K.set_value(modelRNN.optimizer.decay, lr_decay)

    #earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=3, verbose=1)
    #checkpt = ModelCheckpoint(monitor="val_loss", mode="auto", filepath='weights/rnn_model_baseline_weights.hdf5',
    #                          verbose=1,
    #                          save_best_only=True)
    #rlrop = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=1, verbose=1, factor=0.1, cooldown=0,
    #                         min_lr=1e-6)
    # Fit the NN Model
    model.load_weights('./weights/all_nn.hdf5',by_name=True)#, skip_mismatch=True)

    #TODO
    nb_to_train_dense = range(0,4)
    to_train_dense = list(map(lambda x: 'no_dense_%d' % x, nb_to_train_dense))
    nb_to_train_batch_normalization = range(0,3)
    to_train_batch_normalization = list(map(lambda x: 'no_bn_%d' % x, nb_to_train_batch_normalization))

    for layer in model.layers:
        layer.trainable = False
    for name in to_train_dense:
        layer = model.get_layer(name=name)
        layer.trainable = True
    for name in to_train_batch_normalization:
        layer = model.get_layer(name=name)
        layer.trainable = True

    #model.get_layer('load_dense_3').trainable = True

    lr = 0.003
    #TODO: learning rate?
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=root_mean_squared_error)
    model.summary()

    data_seq_train = data_sequence(train, batch_size, zip_path, tokenizer)
    data_seq_val = data_sequence(val, batch_size, zip_path, tokenizer)

    past_best = 0
    # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
    it = 30
    pat = 0
    for i in range(it):
        print('Iteration %d begins' % i)
        model.fit_generator(data_seq_train, steps_per_epoch=200, epochs=1,
                            verbose=0,
                            max_queue_size=10, workers=8, use_multiprocessing=True, )

        hist = model.evaluate_generator(data_seq_val, steps=18,
                                        max_queue_size=12, workers=6, use_multiprocessing=True)
        if i > 0:
            curr_loss = hist
            if curr_loss < past_best:
                model.save_weights('./weights/all_nn_continued.hdf5')
                print('Loss improved from %f to %f!' % (past_best, curr_loss))
                past_best = hist
            else:
                pat += 1
                print('Loss did not improve from %f!' % past_best)
                if pat == 2:
                    if lr < 0.0009:
                        break
                    lr /= 5
                    K.set_value(model.optimizer.lr, lr)
                    pat = 0

        else:
            past_best = hist
        print(hist)
        gc.collect()




def pred():
    test = pd.read_csv('../data/all_nn_test.csv')

    zip_path = '../data/test_jpg.zip'

    with open('./others/own_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    batch_size = 48

    model = RNN_model(False)


    model.load_weights('./weights/all_nn_good.hdf5')

    model.compile(optimizer=optimizers.Adam(lr=0.005),
                  loss=root_mean_squared_error)

    data_seq_pred = data_sequence_preds(test, batch_size, zip_path, tokenizer)

    preds = model.predict_generator(data_seq_pred, max_queue_size=10, workers=10, use_multiprocessing=True, verbose=1)

    scores = pd.DataFrame(preds, index=test['item_id'], columns=['deal_probability'])
    #scores['item_id'] = test['item_id']
    #scores.set_index('item_id')

    scores.to_csv('../data/sub_all_nn2.csv')



def text_and_image_features():
    train = pd.read_csv('../data/all_nn_train.csv', nrows=500000, skiprows=500000)
    train = train[['item_id', 'title_description', 'image']]
    gc.collect()

    zip_path = '../data/train_jpg.zip'

    with open('./others/own_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    batch_size = 64

    model = RNN_model_feat()

    # TODO: does this work?
    model.load_weights('./weights/image_nn_serve_27.hdf5', by_name=True)  # , skip_mismatch=True)
    # TODO especially are wrong layers loaded? does it matter?

    model.load_weights('./weights/rnn_greedy.hdf5', by_name=True)  # , skip_mismatch=True)

    model.compile(optimizer=optimizers.Adam(lr=0.005),
                  loss=root_mean_squared_error)

    data_seq_pred = data_sequence_feat(train, batch_size, zip_path, tokenizer)

    preds = model.predict_generator(data_seq_pred, max_queue_size=10, workers=8, use_multiprocessing=True, verbose=1)
    text = pd.DataFrame(preds[0])
    image = pd.DataFrame(preds[1])
    text['item_id'] = train['item_id']
    image['item_id'] = train['item_id']
    text.to_csv('../data/train_text_feat_2.csv')
    image.to_csv(('../data/train_image_feat_2.csv'))

#K.clear_session()
train_only()
#print('finished training')

gc.collect()
K.clear_session()
for iii in range(20):
    gc.collect()

print('starting predictions')
pred()


#gc.collect()
#K.clear_session()
#for ii in range(20):
#    gc.collect()
#print('starting to extract im and text feat')
#text_and_image_features()