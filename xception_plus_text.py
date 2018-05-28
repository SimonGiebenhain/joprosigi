import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile
import os
from keras import backend as K
from keras.applications.xception import Xception

from keras.layers import concatenate
from keras.utils import Sequence

from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.noise import AlphaDropout, GaussianNoise
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, Add, GlobalAveragePooling2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence

import time
import pickle


#TODO calculate from combined train_df, val_df
nr_user_id                 =  771769 + 1
nr_region                  =      28 + 1
nr_city                    =    1733 + 1
nr_parent_category_name    =       9 + 1
nr_category_name           =      47 + 1
nr_param_1                 =     372 + 1
nr_param_2                 =     272 + 1
nr_param_3                 =    1220 + 1
nr_title                   =  788377 + 1
nr_description             = 1317103 + 1
nr_price                   =  771769 + 1
nr_item_seq_number         =   28232 + 1
nr_user_type               =       3 + 1
nr_image                   = 1390837 + 1
nr_image_top_1             =    3063 + 1
nr_deal_probability        =   18407 + 1
nr_week_day                =       7 + 1
nr_month_day               =      21 + 1

max_features = 100000
maxlen_t = 15
maxlen_d = 60
embed_size = 300
filter_sizes = [1,2,3,4]
num_filters = 32


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('../data/wiki.ru.vec'))


def set_up_tokenizer(train_df, test_df):
    train_t = train_df['title'].values
    train_d = train_df['description'].values
    test_t = test_df['title'].values
    test_d = test_df['description'].values
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_t) + list(test_t))
    tokenizer.fit_on_texts(list(train_d) + list(test_d))
    train_t = tokenizer.texts_to_sequences(train_t)
    test_t = tokenizer.texts_to_sequences(test_t)
    train_d = tokenizer.texts_to_sequences(train_d)
    test_d = tokenizer.texts_to_sequences(test_d)
    train_t = sequence.pad_sequences(train_t, maxlen=maxlen_t)
    test_t = sequence.pad_sequences(test_t, maxlen=maxlen_t)
    train_d = sequence.pad_sequences(train_d, maxlen=maxlen_d)
    test_d = sequence.pad_sequences(test_d, maxlen=maxlen_d)
    train_df['title'] = train_t
    train_df['description'] = train_d
    test_df['title'] = test_t
    test_df['description'] = test_d
    return tokenizer


base_model = Xception(weights='imagenet', include_top=False)
for layer in base_model.layers[:115]:
   layer.trainable = False
for layer in  base_model.layers[115:]:
   layer.trainable = True

#TODO: include more features and deeper densyl connected regression network
#feat_in = Input(shape=(nr_feat,), name='input_2')
x_base = base_model.output
#TODO what does this layer do?
x_base = GlobalAveragePooling2D(name='avg_pool')(x_base)
# let's add a fully-connected layer
x_base = Dense(1024, activation='relu', )(x_base)

def build_model(base_output, tok, lr):
    word_index = tok.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    inp_t = Input(shape=(maxlen_t,))
    inp_d = Input(shape=(maxlen_d,))
    x_t = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp_t)
    x_d = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp_d)
    x_t = SpatialDropout1D(0.3)(x_t)
    x_d = SpatialDropout1D(0.3)(x_d)
    x_t = Reshape((maxlen_t, embed_size, 1))(x_t)
    x_d = Reshape((maxlen_d, embed_size, 1))(x_d)
    conv_0_t = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                      activation='elu')(x_t)
    conv_1_t = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                      activation='elu')(x_t)
    conv_2_t = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                      activation='elu')(x_t)
    conv_3_t = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                      activation='elu')(x_t)
    # conv_4_t = Conv2D(num_filters, kernel_size=(filter_sizes[4], embed_size), kernel_initializer='normal', activation='elu')(x_t)
    conv_0_d = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                      activation='elu')(x_d)
    conv_1_d = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                      activation='elu')(x_d)
    conv_2_d = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                      activation='elu')(x_d)
    conv_3_d = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                      activation='elu')(x_d)
    # conv_4_d = Conv2D(num_filters, kernel_size=(filter_sizes[4], embed_size), kernel_initializer='normal', activation='elu')(x_d)
    maxpool_0_t = MaxPool2D(pool_size=(maxlen_t - filter_sizes[0] + 1, 1))(conv_0_t)
    maxpool_1_t = MaxPool2D(pool_size=(maxlen_t - filter_sizes[1] + 1, 1))(conv_1_t)
    maxpool_2_t = MaxPool2D(pool_size=(maxlen_t - filter_sizes[2] + 1, 1))(conv_2_t)
    maxpool_3_t = MaxPool2D(pool_size=(maxlen_t - filter_sizes[3] + 1, 1))(conv_3_t)
    # maxpool_4_t = MaxPool2D(pool_size=(maxlen_t - filter_sizes[4] + 1, 1))(conv_4_t)
    maxpool_0_d = MaxPool2D(pool_size=(maxlen_d - filter_sizes[0] + 1, 1))(conv_0_d)
    maxpool_1_d = MaxPool2D(pool_size=(maxlen_d - filter_sizes[1] + 1, 1))(conv_1_d)
    maxpool_2_d = MaxPool2D(pool_size=(maxlen_d - filter_sizes[2] + 1, 1))(conv_2_d)
    maxpool_3_d = MaxPool2D(pool_size=(maxlen_d - filter_sizes[3] + 1, 1))(conv_3_d)
    # maxpool_4_d = MaxPool2D(pool_size=(maxlen_d - filter_sizes[4] + 1, 1))(conv_4_d)
    z_t = Concatenate(axis=1)([maxpool_0_t, maxpool_1_t, maxpool_2_t, maxpool_3_t])  # , maxpool_4_t])
    z_d = Concatenate(axis=1)([maxpool_0_d, maxpool_1_d, maxpool_2_d, maxpool_3_d])  # , maxpool_4_d])
    z_t = Flatten()(z_t)
    z_t = Dropout(0.2)(z_t)
    z_d = Flatten()(z_d)
    z_d = Dropout(0.2)(z_d)



    inp_reg = Input(shape=(1,))
    inp_city = Input(shape=(1,))
    inp_cat1 = Input(shape=(1,))
    inp_cat2 = Input(shape=(1,))
    inp_prm1 = Input(shape=(1,))
    inp_prm2 = Input(shape=(1,))
    inp_prm3 = Input(shape=(1,))
    inp_sqnm = Input(shape=(1,))
    inp_usr_type = Input(shape=(1,))
    inp_itype = Input(shape=(1,))
    inp_week_day = Input(shape=(1,))
    inp_month_day = Input(shape=(1,))
    inp_price = Input(shape=(1,))
    nsy_price = GaussianNoise(0.1)(inp_price)

    emb_size = 8
    emb_reg =       Embedding(nr_region, emb_size)(inp_reg)
    emb_city =      Embedding(nr_city, emb_size)(inp_city)
    emb_cat1 =      Embedding(nr_parent_category_name, emb_size)(inp_cat1)
    emb_cat2 =      Embedding(nr_category_name, emb_size)(inp_cat2)
    emb_prm1 =      Embedding(nr_param_1, emb_size)(inp_prm1)
    emb_prm2 =      Embedding(nr_param_2, emb_size)(inp_prm2)
    emb_prm3 =      Embedding(nr_param_3, emb_size)(inp_prm3)
    emb_sqnm =      Embedding(nr_item_seq_number, emb_size)(inp_sqnm)
    #emb_usr_id =   Embedding(+ 1, emb_size)(inp_usr_id)
    emb_usr_type =  Embedding(nr_user_type, emb_size)(inp_usr_type)
    emb_week_day =  Embedding(nr_week_day, emb_size)(inp_week_day)
    emb_month_day = Embedding(nr_month_day, emb_size)(inp_week_day)
    emb_itype =     Embedding(nr_image_top_1, emb_size)(inp_itype)
    x = concatenate([emb_reg, emb_city, emb_cat1, emb_cat2, emb_prm1, emb_prm2, emb_prm3,
                     emb_sqnm, emb_usr_type, emb_week_day, emb_month_day, emb_itype])
    x = Flatten()(x)

    #TODO: also dropout image features?
    x = concatenate([x_base, x])
    x = Dropout(0.5)(x)
    x = concatenate([x, nsy_price, z_d, z_t])  # Do not want to dropout price, its noised up instead.

    #TODO: lecun?
    #TODO: Good architecture?
    x = Dense(1024, activation="selu", kernel_initializer="lecun_normal")(x)
    x = AlphaDropout(0.05)(x)
    #x = Dense(512, activation="selu", kernel_initializer="lecun_normal")(x)
    #x = AlphaDropout(0.05)(x)
    x = Dense(512, activation="selu", kernel_initializer="lecun_normal")(x)
    y = Dense(1, activation="sigmoid")(x)


    model = Model(inputs=[base_model.input, inp_reg, inp_city, inp_cat1, inp_cat2, inp_prm1, inp_prm2,
                          inp_prm3, inp_sqnm, inp_usr_type, inp_itype, inp_week_day, inp_month_day, inp_price, inp_t, inp_d],
                  outputs=y)
    model.compile(optimizer=Adam(lr=lr, clipnorm=0.5), loss=root_mean_squared_error)
    #model.summary()

    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




class data_sequence(Sequence):
    def __init__(self, x_set, batch_size, zip_path):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_1 = np.zeros((self.batch_size, 299, 299, 3), np.float32)
        reg = np.zeros((self.batch_size,), np.float32)
        city = np.zeros((self.batch_size,), np.float32)
        cat1 = np.zeros((self.batch_size,), np.float32)
        cat2 = np.zeros((self.batch_size,), np.float32)
        prm1 = np.zeros((self.batch_size,), np.float32)
        prm2 = np.zeros((self.batch_size,), np.float32)
        prm3 = np.zeros((self.batch_size,), np.float32)
        sqnm = np.zeros((self.batch_size,), np.float32)
        usr_type = np.zeros((self.batch_size,), np.float32)
        itype = np.zeros((self.batch_size,), np.float32)
        week_day = np.zeros((self.batch_size,), np.float32)
        month_day = np.zeros((self.batch_size,), np.float32)
        price = np.zeros((self.batch_size,), np.float32)
        t = np.zeros((self.batch_size, maxlen_t), np.float32)
        d = np.zeros((self.batch_size, maxlen_d), np.float32)
        y = np.zeros(self.batch_size, np.float32)
        i = 0
        with ZipFile(self.zip_path) as im_zip:
            for ad in batch_x.itertuples():
                try:
                    file = im_zip.open(getattr(ad,'image') + '.jpg')
                    img = image.load_img(file, target_size=(299, 299))
                    img = image.img_to_array(img)
                    mean = np.mean(img)
                    std = np.std(img)
                    img = img - mean
                    if std < 0.01 or np.isinf(img).any() or np.isnan(img).any():
                        img = np.zeros((299, 299, 3))
                    x_1[i] = img
                except KeyError:
                    x_1[i] = np.zeros((299,299,3))
                reg[i] = getattr(ad, 'region')
                city[i] = getattr(ad, 'city')
                cat1[i] = getattr(ad, 'parent_category_name')
                cat2[i] = getattr(ad, 'category_name')
                prm1[i] = getattr(ad, 'param_1')
                prm2[i] = getattr(ad, 'param_2')
                prm3[i] = getattr(ad, 'param_3')
                sqnm[i] = getattr(ad, 'item_seq_number')
                usr_type[i] = getattr(ad, 'user_type')
                itype[i] = getattr(ad, 'image_top_1')
                week_day[i] = getattr(ad, 'week_day')
                month_day[i] = getattr(ad, 'month_day')
                price[i] = getattr(ad,'price')
                t[i] = getattr(ad, 'title')
                d[i] = getattr(ad, 'description')
                y[i] = getattr(ad,'deal_probability')
                i = i+1
        return [x_1, reg, city, cat1, cat2, prm1, prm2, prm3, sqnm, usr_type, itype, week_day, month_day, price, t, d], [y]

    def set_x_set(self, x_set):
        self.x = x_set

class data_sequence_test(Sequence):
    def __init__(self, x_set, batch_size, zip_path):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        x_1 = np.zeros((self.batch_size, 299, 299, 3), np.float32)
        reg = np.zeros((self.batch_size,), np.float32)
        city = np.zeros((self.batch_size,), np.float32)
        cat1 = np.zeros((self.batch_size,), np.float32)
        cat2 = np.zeros((self.batch_size,), np.float32)
        prm1 = np.zeros((self.batch_size,), np.float32)
        prm2 = np.zeros((self.batch_size,), np.float32)
        prm3 = np.zeros((self.batch_size,), np.float32)
        sqnm = np.zeros((self.batch_size,), np.float32)
        usr_type = np.zeros((self.batch_size,), np.float32)
        itype = np.zeros((self.batch_size,), np.float32)
        week_day = np.zeros((self.batch_size,), np.float32)
        month_day = np.zeros((self.batch_size,), np.float32)
        price = np.zeros((self.batch_size,), np.float32)
        y = np.zeros(self.batch_size, np.float32)
        i = 0
        with ZipFile(self.zip_path) as im_zip:
            for ad in batch_x.itertuples():
                try:
                    file = im_zip.open(getattr(ad,'image') + '.jpg')
                    img = image.load_img(file, target_size=(299, 299))
                    img = image.img_to_array(img)
                    mean = np.mean(img)
                    std = np.std(img)
                    img = img - mean
                    if std < 0.01 or np.isinf(img).any() or np.isnan(img).any():
                        img = np.zeros((299, 299, 3))
                    x_1[i] = img
                except KeyError:
                    x_1[i] = np.zeros((299,299,3))
                reg[i] = getattr(ad, 'region')
                city[i] = getattr(ad, 'city')
                cat1[i] = getattr(ad, 'parent_category_name')
                cat2[i] = getattr(ad, 'category_name')
                prm1[i] = getattr(ad, 'param_1')
                prm2[i] = getattr(ad, 'param_2')
                prm3[i] = getattr(ad, 'param_3')
                sqnm[i] = getattr(ad, 'item_seq_number')
                usr_type[i] = getattr(ad, 'user_type')
                itype[i] = getattr(ad, 'image_top_1')
                week_day[i] = getattr(ad, 'week_day')
                month_day[i] = getattr(ad, 'month_day')
                price[i] = getattr(ad,'price')
                i = i+1
        return [x_1, reg, city, cat1, cat2, prm1, prm2, prm3, sqnm, usr_type, itype, week_day, month_day, price]


def train():
    batch_size_train = 64
    batch_size_val = 128

    zip_path = '/Users/sigi/.kaggle/competitions/avito-demand-prediction/train_jpg_0.zip'

    print("Reading Data")
    train_df = pd.read_csv("~/PycharmProjects/avito/data/nn_train.csv")
    val_df = pd.read_csv("~/PycharmProjects/avito/data/nn_val.csv")
    print("Finished reading data")
    print('Tokenize')
    tok = set_up_tokenizer(train_df, val_df)
    print('finished tokenizing')

    seq_1 = data_sequence(train_df,batch_size_train, zip_path)
    seq_2 = data_sequence(val_df, batch_size_val, zip_path)

    earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=5, verbose=0)
    checkpt = ModelCheckpoint(monitor="val_loss", mode="auto", filepath='weights/test_model_baseline_weights.hdf5', verbose=0,
                              save_best_only=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=2, verbose=1, factor=0.1, cooldown=0, min_lr=1e-6)

    learning_rate = 0.0001
    model = build_model(x_base, tok, learning_rate)

    prev_loss = 0
    patience = 3
    anger = 0
    #TODO dont use validation sequence, as it continously spwans new workers ----> workaround use callback
    while True:
        if learning_rate < 1e-7:
            break
        model.fit_generator(seq_1, 16, 1,
                            #validation_data=seq_2, validation_steps=2,
                            max_queue_size=8,
                            workers=4, use_multiprocessing=True)
                            #callbacks=[checkpt, earlystop, rlrop])
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        seq_1.set_x_set(train_df)

        #TODO use evaluate instead, use reasonable callbacks, try custom validation callback maybe
        val_loss = model.evaluate_generator(seq_2, steps=5, workers=4, use_multiprocessing=True, verbose=1)
        val_df = val_df.sample(frac=1).reset_index(drop=True)
        seq_2.set_x_set(val_df)
        print(val_loss)
        if prev_loss > 0 and prev_loss <= val_loss:
            anger = anger + 1
        elif prev_loss > val_loss and anger > 0:
            anger = anger - 1
        if anger >= patience:
            learning_rate = learning_rate / 10
            K.set_value(model.optimizer.lr, learning_rate)
            anger = 0
        if prev_loss > val_loss:
            model.save_weights('./weights/xception_best.hdf5')
        prev_loss = val_loss
        model.save_weights('./weights/xception_text.hdf5')

#def predict():
#    model.load_weights('weights/every_epoch.hdf5')
#    batch_size_test = 50
#    zip_path_test = '../data/test_jpg.zip'
#    test_df = pd.read_csv('../data/nn_test.csv')
#    all_test_df = np.array_split(test_df,10)
#    pred_list = []
#    for i in range(10):
#        seq_1 = data_sequence_test(all_test_df[i], batch_size=batch_size_test, zip_path=zip_path_test)
#        preds = model.predict_generator(seq_1, workers=4,use_multiprocessing=True, verbose=1)
#        np.save('../data/nn_predictions_'+ str(i) + '.npy' , preds)
#        pred_list.append(preds)
#    submission = pd.read_csv('../data/nn_sample_submission.csv')
#    submission['deal_probability'] = np.concatenate(pred_list, axis=0)
#    submission.to_csv("submission.csv", index=False)

train()