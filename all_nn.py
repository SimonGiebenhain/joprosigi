import pandas as pd
import numpy as np
import time
import gc
import os
import warnings
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

def add_common_layers(y):
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
            'item_seq_number', 'activation_date', 'image', 'deal_probability']

EMBEDDING_DIM1 = 300  # this is from the pretrained vectors
max_region = 28
max_city = 1752
max_category_name = 47
max_parent_category_name = 9
max_param_1 = 372
max_param123 = 2402
max_image_code = 3064

max_len_text = 100
max_num_words = 200000

#TODO
#fill price with average of category
#treat image_top_1 as numerical, categorical or string?
#use activation weekday'''

def preprocess_dataset(dataset):
    print("Filling Missing Values.....")

    dataset['price'] = dataset['price'].fillna(0).astype('float32')
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

    dataset['image_top_1'] = dataset['image_top_1'].fillna('missing')
    dataset['image_code'] = dataset['image_top_1'].astype('str')
    del dataset['image_top_1']
    gc.collect()

    # dataset['week'] = pd.to_datetime(dataset['activation_date']).dt.week.astype('uint8')
    # dataset['day'] = pd.to_datetime(dataset['activation_date']).dt.day.astype('uint8')
    # dataset['wday'] = pd.to_datetime(dataset['activation_date']).dt.dayofweek.astype('uint8')
    del dataset['activation_date']
    gc.collect()

    print("Creating New Feature.....")
    dataset['param123'] = (dataset['param_1'] + '_' + dataset['param_2'] + '_' + dataset['param_3']).astype(str)
    del dataset['param_2'], dataset['param_3']
    gc.collect()

    dataset['title_description'] = (dataset['title'] + " " + dataset['description']).astype(str)
    del dataset['description'], dataset['title']
    gc.collect()

    dataset['price'] = np.log1p(dataset['price'])
    dataset['avg_days_up_user'] = np.log1p(dataset['avg_days_up_user'])
    dataset['avg_times_up_user'] = np.log1p(dataset['avg_times_up_user'])
    dataset['n_user_items'] = np.log1p(dataset['n_user_items'])
    dataset['item_seq_number'] = np.log(dataset['item_seq_number'])

    dataset['avg_days_up_user'] = dataset['avg_days_up_user'].fillna(0).astype('uint32')
    dataset['avg_times_up_user'] = dataset['avg_times_up_user'].fillna(0).astype('uint32')
    dataset['n_user_items'] = dataset['n_user_items'].fillna(0).astype('uint32')

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
    # le_wday = LabelEncoder()
    # le_wday.fit(DF.wday)
    print("Fit on Train Function completed.")


    df['region'] = le_region.transform(df['region'])
    df['city'] = le_city.transform(df['city'])
    df['category_name'] = le_category_name.transform(df['category_name'])
    df['parent_category_name'] = le_parent_category_name.transform(df['parent_category_name'])
    df['param_1'] = le_param_1.transform(df['param_1'])
    df['param123'] = le_param123.transform(df['param123'])
    # dataset['day'] = le_day.transform(dataset['day'])
    # dataset['week'] = le_week.transform(dataset['week'])
    # dataset['wday'] = le_wday.transform(dataset['wday'])
    df['image_code'] = le_image_code.transform(df['image_code'])

    print("Transform on test function completed.")

    return df


def get_keras_data(dataset, tokenizer):
    t = tokenizer.texts_to_sequences(dataset['title_description'].astype('str'))
    X = {
        'seq_title_description': sequence.pad_sequences(t, maxlen=100),
        'region': dataset.region.values,
        'city': dataset.city.values,
        'category_name': dataset.category_name.values,
        'parent_category_name': dataset.parent_category_name.values,
        'param_1': dataset.param_1.values,
        'param123': dataset.param123.values,
        'image_code': dataset.image_code.values,
        'price': dataset.price.values,
        'item_seq_number': dataset.item_seq_number.values,
        'avg_ad_days': dataset.avg_days_up_user.values,
        'avg_ad_times': dataset.avg_times_up_user.values,
        'n_user_items': dataset.n_user_items.values,
    }
    return X


class data_sequence(Sequence):
    def __init__(self, x_set, batch_size, zip_path, tokenizer):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path
        self.tokenizer = tokenizer

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        image_hashs = batch_x.image.values
        images = np.zeros((self.batch_size, img_height, img_width, img_channels), dtype=np.float32)
        with ZipFile(self.zip_path) as im_zip:
            for i, hash in enumerate(image_hashs):
                try:
                    file = im_zip.open(str(hash) + '.jpg')
                    img = image.load_img(file, target_size=(img_height, img_width))
                    img = image.img_to_array(img)
                    if not (np.isinf(img).any() or np.isnan(img).any()):
                        img = xception.preprocess_input(img)
                    images[i] = img
                except KeyError:
                    print('Error loading image %s' % hash)
                except IOError:
                    print('Error loading image %s' % hash)

        texts = sequence.pad_sequences(self.tokenizer.texts_to_sequences(batch_x.title_description.astype('str')), maxlen=max_len_text)
        X = {
            'image': images,
            'seq_title_description': texts,
            'region': batch_x.region.values,
            'city': batch_x.city.values,
            'category_name': batch_x.category_name.values,
            'parent_category_name': batch_x.parent_category_name.values,
            'param_1': batch_x.param_1.values,
            'param123': batch_x.param123.values,
            'image_code': batch_x.image_code.values,
            #TODO: include: 'user_type': batch_x.user_type.values,
            'price': batch_x.price.values,
            'item_seq_number': batch_x.item_seq_number.values,
            'avg_ad_days': batch_x.avg_days_up_user.values,
            'avg_ad_times': batch_x.avg_times_up_user.values,
            'n_user_items': batch_x.n_user_items.values
        }
        Y = batch_x.deal_probability.values
        return X, Y


    def set_x_set(self, x_set):
        self.x = x_set


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

    with open('./others/tokenizer.pickle', 'wb') as handle:
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
    np.save('./others/emb_mat_test.npy', embedding_matrix1)
    del embedding_matrix1
    gc.collect()
    #TODO hard code vocab size
    print('Vocab size: ' + str(vocab_size))

    train = all[:train_size]
    train.to_csv('../data/rnn_train.csv')
    del train
    gc.collect()
    test = all[train_size:]
    test.to_csv('../data/rnn_test.csv')
    del test
    del all
    gc.collect()
    np.save('../data/rnn_y_train.npy', y_train)



img_height = 224
img_width = 224
img_channels = 3

def residual_network(x):

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):

        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, name='load_dense_1')(x)
    x = add_common_layers(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, name='load_dense_2')(x)
    x = add_common_layers(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, name='load_dense_3')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    #x = layers.Dense(1, activation=activations.sigmoid)(x)

    return x




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

    image_input = layers.Input(shape=(img_height, img_width, img_channels), name='image')
    image_output = residual_network(image_input)

    seq_title_description = Input(shape=[100], name="seq_title_description")

    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")
    param_1 = Input(shape=[1], name="param_1")
    param123 = Input(shape=[1], name="param123")
    image_code = Input(shape=[1], name="image_code")
    price = Input(shape=[1], name="price")
    item_seq_number = Input(shape=[1], name='item_seq_number')
    avg_ad_days = Input(shape=[1], name="avg_ad_days")
    avg_ad_times = Input(shape=[1], name="avg_ad_times")
    n_user_items = Input(shape=[1], name="n_user_items")

    emb_size_small = 6
    emb_size_big = 14
    # Embeddings layers
    embedding_matrix1 = np.load('./others/emb_mat.npy')
    vocab_size = embedding_matrix1.shape[0]
    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights=[embedding_matrix1], trainable=False)(
        seq_title_description)
    emb_region = Embedding(max_region, emb_size_small, name='emb_region')(region)
    emb_city = Embedding(max_city, emb_size_small, name='emb_city')(city)
    emb_category_name = Embedding(max_category_name, emb_size_small, name='emb_category')(category_name)
    emb_parent_category_name = Embedding(max_parent_category_name, emb_size_small, name='emb_parent_category')(
        parent_category_name)
    emb_param_1 = Embedding(max_param_1, emb_size_big, name='emb_param1')(param_1)
    emb_param123 = Embedding(max_param123, emb_size_big, name='emb_param123')(param123)
    emb_image_code = Embedding(max_image_code, emb_size_big, name='emb_image_code')(image_code)

    rnn_layer1 = GRU(50, name='GRU_load')(emb_seq_title_description)

    # main layer
    cat_net = concatenate([ Flatten()(emb_region),
                            Flatten()(emb_city),
                            Flatten()(emb_category_name),
                            Flatten()(emb_parent_category_name),
                            Flatten()(emb_param_1),
                            Flatten()(emb_param123),
                            Flatten()(emb_image_code),
                            ])
    cat_net = Dense(32, name='dense_no_load_1')(Dropout(0.1)(cat_net))
    cat_net = add_common_layers(cat_net)
    cat_net = Dense(16, name='dense_no_load_2')(Dropout(0.1)(cat_net))
    cat_net = add_common_layers(cat_net)
    main_l = concatenate([image_output,
                          rnn_layer1,
                          cat_net])
    main_l = Dense(100, name='dense_no_load_3')(Dropout(0.1)(main_l))
    main_l = add_common_layers(main_l)
    main_l = Dense(50, name='dense_no_load_4')(Dropout(0.1)(main_l))
    main_l = add_common_layers(main_l)
    all = concatenate([main_l,
                       avg_ad_days,
                       avg_ad_times,
                       n_user_items,
                       price,
                       item_seq_number])
    # output
    all = Dense(32, name='dense_no_load_5')(all)
    all = add_common_layers(all)
    output = Dense(1, activation="sigmoid", name='dense_no_load_6')(all)

    # model
    model = Model([image_input, seq_title_description, region, city, category_name, parent_category_name,
                   param_1, param123, image_code, price, item_seq_number,
                   avg_ad_days, avg_ad_times, n_user_items], output)
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


#TODO: Check whether numerical data is properly normalized!
#Check wheterher order of train and y_train corresponds!
def train_only():
    train = pd.read_csv('../data/rnn_train.csv')
    y_train = np.load('../data/rnn_y_train.npy')
    train['deal_probability'] = y_train

    zip_path = '../data/train_jpg_0.zip'
    files_in_zip = ZipFile(zip_path).namelist()
    item_ids = list(map(lambda s: os.path.splitext(s)[0], files_in_zip))
    train = train.loc[train['image'].isin(item_ids)]

    train = train.sample(frac=1).reset_index(drop=True)
    train_len = int(len(train.index) * 0.33)
    val = train[-train_len:]
    train = train[:-train_len]


    with open('./others/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Initialize a new Model for Current FOLD
    batch_size = 64
    #steps = (int(train.shape[0] / batch_size)) * epochs
    #lr_init, lr_fin = 0.009, 0.00045
    #lr_decay = exp_decay(lr_init, lr_fin, steps)
    model = RNN_model()
    #K.set_value(modelRNN.optimizer.decay, lr_decay)

    #earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=3, verbose=1)
    #checkpt = ModelCheckpoint(monitor="val_loss", mode="auto", filepath='weights/rnn_model_baseline_weights.hdf5',
    #                          verbose=1,
    #                          save_best_only=True)
    #rlrop = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=1, verbose=1, factor=0.1, cooldown=0,
    #                         min_lr=1e-6)
    # Fit the NN Model
    #TODO: does this work?
    model.load_weights('./weights/image_nn.hdf5',by_name=True, skip_mismatch=True)
    #TODO especially are wrong layers loaded? does it matter?
    model.load_weights('./weights/rnn_train_only_best_2nd.hdf5', by_name=True, skip_mismatch=True)
    #gru_weights = np.load('./weights/gru.npy')
    #model.get_layer('GRU').set_weights(gru_weights)

    nb_to_train_dense = range(1,7)
    to_train_dense = list(map(lambda x: 'dense_no_load_%d' % x, nb_to_train_dense))
    nb_to_train_batch_normalization = range(27,32)
    to_train_batch_normalization = list(map(lambda x: 'batch_normalization_%d' % x, nb_to_train_batch_normalization))

    for layer in model.layers:
        layer.trainable = False
    for name in to_train_dense:
        layer = model.get_layer(name=name)
        layer.trainable = True
    for name in to_train_batch_normalization:
        layer = model.get_layer(name=name)
        layer.trainable = True

    #model.get_layer('load_dense_3').trainable = True

    #TODO: learning rate?
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=root_mean_squared_error,
                  metrics=[root_mean_squared_error])
    model.summary()

    data_seq_train = data_sequence(train, batch_size, zip_path, tokenizer)
    data_seq_val = data_sequence(val, batch_size, zip_path, tokenizer)

    past_best = 0
    # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
    for i in range(50):
        print('Iteration %d begins' % i)
        hist = model.fit_generator(data_seq_train, steps_per_epoch=50, epochs=1,
                                   verbose=1,
                                   validation_data=data_seq_val, validation_steps=10,
                                   max_queue_size=8, workers=4, use_multiprocessing=True)
        train = train.sample(frac=1).reset_index(drop=True)
        data_seq_train.set_x_set(train)
        # model.evaluate_generator(data_seq_val,steps=1)
        if i > 0:
            curr_loss = hist.history['val_loss'][-1]
            if curr_loss < past_best:
                model.save_weights('./weights/all_nn_2nd.hdf5')
                print('Loss improved from %f to %f!' % (past_best, curr_loss))
                past_best = hist.history['val_loss'][-1]
            else:
                print('Loss did not improve from %f!' % past_best)
                #TODO decrease learning rate, plus patience

        else:
            past_best = hist.history['val_loss'][-1]

        gc.collect()

    # Predict test set for Kth Fold
    model.save_weights('./weights/rnn_train_only.hdf5')


###############
#TODO: include wday

###############

train_only()