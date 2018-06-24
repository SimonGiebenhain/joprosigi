import pandas as pd
import numpy as np
import time
import gc
import os
import warnings
import pickle
import string
import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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
from keras import layers


from keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
cores = 4

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    return y

exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1


dtypes_train = {
                'price': 'float32',
                'item_seq_number': 'uint32',
                'deal_probability': 'float32'
}
dtypes_test = {
                'price': 'float32',
                'item_seq_number': 'uint32'
}
use_cols = ['item_id', 'user_id', 'image_top_1', 'region', 'city', 'parent_category_name',
            'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description', 'price',
            'item_seq_number', 'activation_date', 'user_type', 'deal_probability']

EMBEDDING_DIM1 = 300  # this is from the pretrained vectors
max_region = 28
#max_city = 1752 without region city concat
max_city = 1825
max_category_name = 47
max_parent_category_name = 9
max_param_1 = 372
max_param123 = 2402
max_image_code = 3064
max_wday = 32
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
    dataset['wday'] = pd.to_datetime(dataset['activation_date']).dt.day
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


    #TODO: NA for aggregated features: indicator plus mean
    dataset['no_avg_days_up_user'] = dataset['avg_days_up_user'].isna()
    #dataset['avg_days_up_user'].fillna(dataset.avg_days_up_user.mean(), inplace=True)

    dataset['no_avg_times_up_user'] = dataset['avg_times_up_user'].isna()
    #dataset['avg_times_up_user'].fillna(dataset.avg_times_up_user.mean(), inplace=True)

    dataset['no_n_user_items'] =  dataset['n_user_items'].isna()
    #dataset['n_user_items'].fillna(dataset.n_user_items.mean(), inplace=True)

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
    del dataset['param_2'], dataset['param_3']
    gc.collect()

    dataset['title_description'] = (dataset['title'] + " " + dataset['description']).astype(str)
    del dataset['description'], dataset['title']
    gc.collect()

    #Indicator column wheter price was present or not
    #dataset['no_price'] = dataset['price'].isna()
    #Replace NaNs in price with mean price for that category
    #mean_price_df = dataset.groupby('category_name', as_index=False)['price'].mean()
    #mean_price_df['mean_price'] = mean_price_df['price']
    #print(mean_price_df.head())
    #del mean_price_df['price']
    #dataset = dataset.merge(mean_price_df, on=['category_name'], how='left')
    #dataset['price'] = dataset['price'].fillna(dataset['mean_price'])
    #del dataset['mean_price']
    dataset['price'] = dataset['price'].fillna(0)


    #First scale with log1p, bc. this is more natural for humans, then Z-transform
    dataset['price'] = np.log1p(dataset['price'])
    #dataset['price'] = ztransform(dataset.price.values)

    dataset['avg_days_up_user'] = np.log1p(dataset['avg_days_up_user'])
    #dataset['avg_days_up_user'] = ztransform(dataset.avg_days_up_user.values)

    dataset['avg_times_up_user'] = np.log1p(dataset['avg_times_up_user'])
    #dataset['avg_times_up_user'] = ztransform(dataset.avg_times_up_user.values)

    dataset['n_user_items'] = np.log1p(dataset['n_user_items'])
    #dataset['n_user_items'] = ztransform(dataset.n_user_items.values)

    dataset['item_seq_number'] = np.log1p(dataset['item_seq_number'])
    #dataset['item_seq_number'] = ztransform(dataset.item_seq_number.values)

    dataset['avg_days_up_user'] = dataset['avg_days_up_user'].fillna(-1)
    dataset['avg_times_up_user'] = dataset['avg_times_up_user'].fillna(-1)
    dataset['n_user_items'] = dataset['n_user_items'].fillna(-1)

    print("PreProcessing Function completed.")

    return dataset


def set_up_tokenizer(df):
    t = df['title_description'].values
    tokenizer = text.Tokenizer(num_words=200000)
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
    # le_day = LabelEncoder()
    # le_day.fit(DF.day)
    le_wday = LabelEncoder()
    le_wday.fit(df.wday)

    le_user_type = LabelEncoder()
    le_user_type.fit(df.user_type)

    print("Fit on Train Function completed.")


    df['region'] = le_region.transform(df['region'])
    df['city'] = le_city.transform(df['city'])
    df['category_name'] = le_category_name.transform(df['category_name'])
    df['parent_category_name'] = le_parent_category_name.transform(df['parent_category_name'])
    df['param_1'] = le_param_1.transform(df['param_1'])
    df['param123'] = le_param123.transform(df['param123'])
    # dataset['day'] = le_day.transform(dataset['day'])
    # dataset['week'] = le_week.transform(dataset['week'])
    df['wday'] = le_wday.transform(df['wday'])
    df['image_code'] = le_image_code.transform(df['image_code'])
    df['user_type'] = le_user_type.transform(df['user_type'])

    print("Transform on test function completed.")

    return df


def get_keras_data(dataset, tokenizer):
    t = tokenizer.texts_to_sequences(dataset['title_description'].astype('str'))
    X = {
        'seq_title_description': sequence.pad_sequences(t, maxlen=100),
        'region': dataset.region.values,
        'city': dataset.city.values,
        'wday': dataset.wday.values,
        'user_type': dataset.user_type.values,
        #'no_price': dataset.no_price.values,
        'category_name': dataset.category_name.values,
        'parent_category_name': dataset.parent_category_name.values,
        'param_1': dataset.param_1.values,
        'param123': dataset.param123.values,
        'image_code': dataset.image_code.values,
        'price': dataset.price.values,
        'item_seq_number': dataset.item_seq_number.values,
        'avg_ad_days': dataset.avg_days_up_user.values,
        'no_avg_ad_days': dataset.no_avg_days_up_user.values,
        'avg_ad_times': dataset.avg_times_up_user.values,
        'no_avg_ad_times': dataset.no_avg_times_up_user.values,
        'n_user_items': dataset.n_user_items.values,
        'no_n_user_items': dataset.no_n_user_items.values,
        #'num_words_title': dataset.num_words_title.values,
        #'num_words_description':dataset.num_words_description.values,
        #'num_unique_words_title':dataset.num_unique_words_title.values,
        #'num_unique_words_description': dataset.num_unique_words_description.values,
        #'words_vs_unique_title': dataset.words_vs_unique_title.values,
        #'words_vs_unique_description': dataset.words_vs_unique_description.values,
        #'num_desc_punct': dataset.num_desc_punct.values
    }
    return X

# TODO
# What about test data when calculating max value?
# Why +2??
# max_seq_title_description_length
# max_words_title_description


def prep_df():
    train = pd.read_csv("../data/train.csv", parse_dates=["activation_date"], usecols = use_cols, dtype = dtypes_train)
    test = pd.read_csv("../data/test.csv", parse_dates=["activation_date"], usecols=use_cols[:-1], dtype=dtypes_test)

    y_train = np.array(train['deal_probability'])
    del train['deal_probability']
    gc.collect()

    train_size = train.shape[0]
    all = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()

    train_features = pd.read_csv('../data/aggregated_features.csv')
    all = all.merge(train_features, on=['user_id'], how='left')
    del train_features
    del all['item_id'], all['user_id']
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

    with open('./others/new_tokenizer.pickle', 'wb') as handle:
       pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO
    # vocabsize + 2
    # dafuq is qoing on with c,c1,w_Y, w_No: WHY?

    EMBEDDING_FILE1 = '../data/cc.ru.300.vec'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
    print(embedding_matrix1.shape)
    # Creating Embedding matrix
    c = 0
    c1 = 0
    w_Y = []
    w_No = []
    for word, i in tokenizer.word_index.items():
        if word in embeddings_index1:
            c += 1
            embedding_vector = embeddings_index1[word]
            w_Y.append(word)
        else:
            embedding_vector = None
            w_No.append(word)
            c1 += 1
        if embedding_vector is not None:
            embedding_matrix1[i] = embedding_vector
#
    print(c, c1, len(w_No), len(w_Y))
    print(embedding_matrix1.shape)
    del embeddings_index1
    gc.collect()

    print(" FAST TEXT DONE")
    np.save('./others/new_emb_mat_test.npy', embedding_matrix1)
    del embedding_matrix1
    gc.collect()
    #TODO hard code vocab size
    print('Vocab size: ' + str(vocab_size))

    train = all[:train_size]
    train.to_csv('../data/new_rnn_train_noz.csv')
    del train
    gc.collect()
    test = all[train_size:]
    test.to_csv('../data/new_rnn_test_noz.csv')
    del test
    del all
    gc.collect()
    np.save('../data/new_rnn_y_train_noz.npy', y_train)

# TODO
# Tune architecture
# initialization of weights
# dont hard code input of seq length for test data!
# TODO MODEL SUMMARY
# TODO Can you build the model without padding the text?
# Bidirection GRU, batch normalization etc.
# How to design fully connected network
# also include convolutions?
# separate title and description?
def RNN_model():
    # Inputs
    seq_title_description = Input(shape=[100], name="seq_title_description")
    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    wday = Input(shape=[1], name="wday")
    user_type = Input(shape=[1], name="user_type")
    #no_price = Input(shape=[1], name="no_price")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")
    param_1 = Input(shape=[1], name="param_1")
    param123 = Input(shape=[1], name="param123")
    image_code = Input(shape=[1], name="image_code")
    price = Input(shape=[1], name="price")
    item_seq_number = Input(shape=[1], name='item_seq_number')
    avg_ad_days = Input(shape=[1], name="avg_ad_days")
    no_avg_ad_days = Input(shape=[1], name="no_avg_ad_days")
    avg_ad_times = Input(shape=[1], name="avg_ad_times")
    no_avg_ad_times = Input(shape=[1], name="no_avg_ad_times")
    n_user_items = Input(shape=[1], name="n_user_items")
    no_n_user_items = Input(shape=[1], name="no_n_user_items")
    #num_words_title = Input(shape=[1], name="num_words_title")
    #num_words_description = Input(shape=[1], name="num_words_description")
    #num_unique_words_title = Input(shape=[1], name="num_unique_words_title")
    #num_unique_words_description = Input(shape=[1], name="num_unique_words_description")
    #words_vs_unique_title = Input(shape=[1], name="words_vs_unique_title")
    #words_vs_unique_description = Input(shape=[1], name="words_vs_unique_description")
    #num_desc_punct = Input(shape=[1], name="num_desc_punct")


    emb_size_small = 6
    emb_size_big = 14
    # Embeddings layers
    embedding_matrix1 = np.load('./others/new_emb_mat_test.npy')
    vocab_size = embedding_matrix1.shape[0]
    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights=[embedding_matrix1], trainable=False)(
        seq_title_description)
    emb_region = Embedding(max_region, emb_size_small, name='emb_region')(region)
    emb_city = Embedding(max_city, emb_size_small, name='emb_city')(city)
    emb_wday = Embedding(max_wday, emb_size_small, name='emb_wday')(wday)
    emb_user_type = Embedding(max_user_type, 4, name='emb_user_type')(user_type)
    emb_category_name = Embedding(max_category_name, emb_size_small, name='emb_category')(category_name)
    emb_parent_category_name = Embedding(max_parent_category_name, emb_size_small, name='emb_parent_category')(parent_category_name)
    emb_param_1 = Embedding(max_param_1, emb_size_big, name='emb_param1')(param_1)
    emb_param123 = Embedding(max_param123, emb_size_big, name='emb_param123')(param123)
    emb_image_code = Embedding(max_image_code, emb_size_big, name='emb_image_code')(image_code)

    #x = SpatialDropout1D(0.2)(emb_seq_title_description)
    #x = Bidirectional(GRU(50, return_sequences=True, dropout=0.05, recurrent_dropout=0.05))(x)
    #x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
    #x = add_common_layers(x)
    #avg_pool = GlobalAveragePooling1D()(x)
    #max_pool = GlobalMaxPooling1D()(x)
    #x = concatenate([avg_pool, max_pool])
    x = GRU(50)(emb_seq_title_description)

    # main layer
    cat_net = concatenate([
        Flatten()(emb_region),
        Flatten()(emb_city),
        Flatten()(emb_wday),
        Flatten()(emb_user_type),
        #no_price,
        no_avg_ad_days,
        no_avg_ad_times,
        no_n_user_items,
        Flatten()(emb_category_name),
        Flatten()(emb_parent_category_name),
        Flatten()(emb_param_1),
        Flatten()(emb_param123),
        Flatten()(emb_image_code),
    ])
    #TODO: Add batch norm before activation?
    cat_net = Dense(128)(Dropout(0.25)(cat_net))
    cat_net = add_common_layers(cat_net)
    cat_net = Dense(32)(Dropout(0.15)(cat_net))
    cat_net = add_common_layers(cat_net)
    cont_net = concatenate([cat_net,
                            avg_ad_days,
                            avg_ad_times,
                            n_user_items,
                            price,
                            item_seq_number])
                            #num_words_title,
                            #num_words_description,
                            #num_unique_words_title,
                            #num_unique_words_description,
                            #words_vs_unique_title,
                            #words_vs_unique_description,
                            #num_desc_punct])
    cont_net = Dropout(0.1)(Dense(64)(cont_net))
    cont_net = add_common_layers(cont_net)
    cont_net = Dense(32)(cont_net)
    cont_net = add_common_layers(cont_net)
    main_net = concatenate([x, cat_net, cont_net])
    main_net = Dense(128)(main_net)
    main_net = add_common_layers(main_net)
    # output
    output = Dense(1, activation="sigmoid")(main_net)

    # model
    #TODO: noprice after user_type
    model = Model([seq_title_description, region, city, wday, user_type, category_name, parent_category_name,
                   param_1, param123, image_code, price, item_seq_number,
                   avg_ad_days, no_avg_ad_days, avg_ad_times, no_avg_ad_times, n_user_items, no_n_user_items], output)
                   #num_words_title,num_words_description, num_unique_words_title, num_unique_words_description,
                   #words_vs_unique_title, words_vs_unique_description, num_desc_punct], output)
    model.compile(optimizer='adam',
                  loss=root_mean_squared_error,
                  metrics=[root_mean_squared_error])
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


def get_data_frame(dataset):
    DF = pd.DataFrame()

    DF['avg_days_up_user'] = np.array(dataset[:, 0])
    DF['avg_times_up_user'] = np.array(dataset[:, 1])
    DF['category_name'] = np.array(dataset[:, 2])
    DF['city'] = np.array(dataset[:, 3])
    DF['image_code'] = np.array(dataset[:, 4])
    DF['item_seq_number'] = np.array(dataset[:, 5])
    DF['n_user_items'] = np.array(dataset[:, 6])
    DF['param123'] = np.array(dataset[:, 7])
    DF['param_1'] = np.array(dataset[:, 8])
    DF['parent_category_name'] = np.array(dataset[:, 9])
    DF['price'] = np.array(dataset[:, 10])
    DF['region'] = np.array(dataset[:, 11])
    DF['title_description'] = np.array(dataset[:, 12])

    return DF


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


def train_only():
    train = pd.read_csv('../data/new_rnn_train_noz.csv')
    y_train = np.load('../data/new_rnn_y_train_noz.npy')
    train['deal_probability'] = y_train
    #train = train.sample(frac=0.1).reset_index(drop=True)
    y_train = train['deal_probability']
    del train['deal_probability']
    gc.collect()

    with open('./others/new_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Initialize a new Model for Current FOLD
    epochs = 5
    batch_size = 512 * 2
    steps = (int(train.shape[0] / batch_size)) * epochs
    lr_init, lr_fin = 0.009, 0.00045
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    modelRNN = RNN_model()
    K.set_value(modelRNN.optimizer.lr, lr_init)
    K.set_value(modelRNN.optimizer.decay, lr_decay)

    #train = train.sample(frac=1).reset_index(drop=True)
    X_train = get_keras_data(train, tokenizer)

    # Fit the NN Model

    # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
    modelRNN.fit(X_train, y_train,
                 batch_size=batch_size, epochs=epochs,
                 validation_split=0.33,
                 shuffle=True,
                 verbose=1)
    # Predict test set for Kth Fold
    modelRNN.save_weights('./weights/new_rnn_with_add_feat_noz.hdf5')


def evaluate_architecture():
    train = pd.read_csv('../data/rnn_train.csv')
    y_train = np.load('../data/rnn_y_train.npy')
    train['deal_probability'] = y_train
    y_train = train['deal_probability']
    del train['deal_probability']
    gc.collect()
    params = [[33,0,0],[33,1,0],[33,2,0],[99,0,0],[99,1,0],[99,2,0],[33,0,1],[33,1,1],[33,2,1],[99,0,1],[99,1,1],[99,2,1]]
    results = []
    for param in params:
        gru_size = param[0]
        gru = param[1]
        big = param[2]
        skf = KFold(n_splits=3)
        for train_idx, test_idx in skf.split(train, y_train):

            with open('./others/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            # Initialize a new Model for Current FOLD
            epochs = 3
            batch_size = 512 * 2
            steps = (int(train.shape[0] / batch_size)) * epochs
            lr_init, lr_fin = 0.009, 0.0005
            lr_decay = exp_decay(lr_init, lr_fin, steps)
            modelRNN = RNN_model(int(gru_size), gru, big)
            K.set_value(modelRNN.optimizer.lr, lr_init)
            K.set_value(modelRNN.optimizer.decay, lr_decay)

            X_train, X_val = train.iloc[train_idx], train.iloc[test_idx]
            print(X_train.shape, X_val.shape)
            Y_train, Y_val = y_train[train_idx], y_train[test_idx]

            X_train = get_keras_data(X_train, tokenizer)
            X_val = get_keras_data(X_val, tokenizer)

            # Fit the NN Model

            # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
            hist = modelRNN.fit(X_train, Y_train,
                         batch_size=batch_size, epochs=3,
                         validation_data=(X_val, Y_val),
                         shuffle=True,
                         verbose=1)
            val_loss = hist.history['val_loss'][-1]
            results.append(val_loss)
            print(results)


train_only()