import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile
import os
from keras import backend as K
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from keras.layers import Input
from keras.layers import concatenate
from keras.utils import Sequence

from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.noise import AlphaDropout, GaussianNoise
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import time
import pickle


#TODO calculate from combined train_df, val_df
nr_user_id                 = 771769+1
nr_region                  =     28+1
nr_city                    =   1733+1
nr_parent_category_name    =      9+1
nr_category_name           =     47+1
nr_param_1                 =    372+1
nr_param_2                 =    272+1
nr_param_3                 =   1220+1
nr_title                   = 788377+1
nr_description             =1317103+1
nr_price                   = 771769+1
nr_item_seq_number         =  28232+1
nr_user_type               =      3+1
nr_image                   =1390837+1
nr_image_top_1             =   3063+1
nr_deal_probability        =  18407+1
nr_week_day                =      7+1
nr_month_day               =     21+1



#TODO: REPLACE WITH KERAS.SEQUENCE!!!!!
#TODO: Incorporate and preprocess remaining features
#TODO: Text feature

#def generate_data_train():
#    x_1 = np.zeros((batch_size, 299, 299,3), np.float32)
#    x_2 = np.zeros((batch_size,nr_feat), np.float32)
#    y = np.zeros((batch_size), np.float32)
#    while True:
#        for k in range(batch_size):
#            ad = train_df.iloc[np.random.randint(0, len(train_Y) - 1)]
#            with ZipFile(zip_path) as im_zip:
#                with im_zip.open(ad['image'] + '.jpg') as file:
#                    #TODO: Maybe preprocessing, especially if no batch_normalization in ResNet
#                    img = image.load_img(file, target_size=(299, 299))
#                    img= image.img_to_array(img)
#                    mean = np.mean(img)
#                    std = np.std(img)
#                    img = img - mean
#                    if std < 0.01 or np.isinf(img).any() or np.isnan(img).any():
#                        img = np.zeros((299,299,3))
#                    x_1[k] = img
#                    x_2[k] = ad['price']
#                    y[k] = ad['deal_probability']
#        yield ({'input_1': x_1, 'input_2': x_2},{'pred': y})


# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:


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

def build_model(base_output):
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
    x = concatenate([x, nsy_price])  # Do not want to dropout price, its noised up instead.

    #TODO: lecun?
    #TODO: Good architecture?
    x = Dense(1024, activation="selu", kernel_initializer="lecun_normal")(x)
    x = AlphaDropout(0.05)(x)
    x = Dense(512, activation="selu", kernel_initializer="lecun_normal")(x)
    x = AlphaDropout(0.05)(x)
    x = Dense(128, activation="selu", kernel_initializer="lecun_normal")(x)
    y = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[base_model.input, inp_reg, inp_city, inp_cat1, inp_cat2, inp_prm1, inp_prm2,
                          inp_prm3, inp_sqnm, inp_usr_type, inp_itype, inp_week_day, inp_month_day, inp_price],
                  outputs=y)
    model.compile(optimizer=Adam(lr=0.00001, clipnorm=0.5), loss=root_mean_squared_error)
    #model.summary()

    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


model = build_model(x_base)


class data_sequence(Sequence):
    def __init__(self, x_set, batch_size, zip_path):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if (idx * self.batch_size > self.__len__() - 51):
            print('WRONG INDEX?!')
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        #if self.is_val:
        #    print('val index: %d' % idx)
        #else:
        #    print('train idx: %d' % idx)
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
                    file = im_zip.open(getattr(ad,'image') + '.yjpg')
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
                y[i] = getattr(ad,'deal_probability')
                i = i+1
        return [x_1, reg, city, cat1, cat2, prm1, prm2, prm3, sqnm, usr_type, itype, week_day, month_day, price], [y]

class data_sequence_test(Sequence):
    def __init__(self, x_set, batch_size, zip_path):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if (idx * self.batch_size > self.__len__() - 51):
            print('WRONG INDEX?!')
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        #if self.is_val:
        #    print('val index: %d' % idx)
        #else:
        #    print('train idx: %d' % idx)
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
    batch_size_train = 50
    # Generate list of items, split into training and validation sets.
    zip_path = '/Users/sigi/.kaggle/competitions/avito-demand-prediction/train_jpg_0.zip'
    files_in_zip = ZipFile(zip_path).namelist()
    item_ids = list(map(lambda s: os.path.splitext(s)[0], files_in_zip))

    print("Reading Data")
    train_df = pd.read_csv("~/PycharmProjects/avito/data/nn_train.csv")
    # val_df = pd.read_csv("~/PycharmProjects/avito/data/nn_val.csv")
    # print(train_df.head)


    ## read
    # file_path = "../data/val_data_nn.pkl"
    # n_bytes = 2**31
    # max_bytes = 2**31 - 1
    # bytes_in = bytearray(0)
    # input_size = os.path.getsize(file_path)
    # with open(file_path, 'rb') as f_in:
    #    for _ in range(0, input_size, max_bytes):
    #        bytes_in += f_in.read(max_bytes)
    # val_data = pickle.loads(bytes_in)

    del train_df['title'], train_df['description']  # , val_df['title'], val_df['description']

    # TODO evalute on test data
    # train_df = pd.read_csv("~/PycharmProjects/avito/data/nn_train.csv")
    print("Finished reading data")

    # Reduce training data to ads with image
    train_df = train_df.loc[train_df['image'].isin(item_ids)]
    # val_df = val_df.loc[val_df['image'].isin(item_ids)]
    train_Y = train_df['deal_probability'].values
    # val_Y = val_df['deal_probability'].values

    print(train_Y.shape)
    # print(val_Y.shape)

    with open('../data/val_data_nn', 'rb') as fp:
        val_data = pickle.load(fp)
    nr_batches = len(val_data)
    seq_1 = data_sequence(train_df,batch_size_train, zip_path)
    #seq_2 = data_sequence(val_df, val_Y, batch_size_val, zip_path, nr_feat, True)

    earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=5, verbose=0)
    checkpt = ModelCheckpoint(monitor="val_loss", mode="auto", filepath='weights/test_model_baseline_weights.hdf5', verbose=0,
                              save_best_only=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=2, verbose=1, factor=0.1, cooldown=0, min_lr=1e-6)

    #TODO dont use validation sequence, as it continously spwans new workers ----> workaround use callback
    #t = time.time()
    val_loss = 0
    while True:
        model.fit_generator(seq_1, 25, 1,
                            #validation_data=seq_2, validation_steps=2,
                            max_queue_size=8,
                            workers=4, use_multiprocessing=True)
                            #callbacks=[checkpt, earlystop, rlrop])
        #TODO use evaluate instead, use reasonable callbacks, try custom validation callback maybe
        cum_loss = 0
        nr_val = 10
        for _ in range(nr_val):
            i = np.random.randint(0,nr_batches - 1)
            l = model.test_on_batch(val_data[i][0], val_data[i][1])
            print('Val loss for batch %d: %f' % (i,l))
            cum_loss = cum_loss + l
        cum_loss = cum_loss / nr_val
        print('average val_loss: %f '% cum_loss)
        model.save_weights('weights/every_epoch.hdf5')
        if val_loss > cum_loss:
            model.save_weights('weights/best_epoch.hdf5')
        val_loss = cum_loss
    #elapsed = time.time() - t
    #print(elapsed)


def continue_train():
    #model.load_weights('weights/model_baseline_weights.hdf5')
    #seq_1 = data_sequence(train_df, train_Y, batch_size_train, zip_path, nr_feat)
    #seq_2 = data_sequence(val_df, val_Y, batch_size_val, zip_path, nr_feat)

    earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=5, verbose=0)
    checkpt = ModelCheckpoint(monitor="val_loss", mode="auto", filepath='weights/model_baseline_weights.hdf5',
                              verbose=0,
                              save_best_only=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=2, verbose=1, factor=0.1, cooldown=0,
                              min_lr=1e-6)

    # TODO Make train, resume train and test method
    #model.fit_generator(seq_1, 100, 50,
    #                    validation_data=seq_2, validation_steps=100,
    #                    workers=2, use_multiprocessing=True, shuffle=True,
    #                    callbacks=[checkpt, earlystop, rlrop])


def eval_model():
    model.load_weights('weights/model_baseline_weights.hdf5')
    #seq_2 = data_sequence(val_df, val_Y, batch_size_val, zip_path, nr_feat, is_val=True)

    #model.evaluate_generator(seq_2, steps=4,workers=4,use_multiprocessing=True, verbose=1)


def test():
    # TODO: Load data all zeros?!?!
    with open('../data/val_data_nn', 'rb') as fp:
        val_data = pickle.load(fp)
    nr_batches = len(val_data)
    model.load_weights('weights/every_epoch.hdf5')
    cum_loss = 0
    nr_val = nr_batches
    for i in range(nr_val):
        #i = np.random.randint(0, nr_batches - 1)
        l = model.test_on_batch(val_data[i][0], val_data[i][1])
        print('Val loss for batch %d: %f' % (i, l))
        cum_loss = cum_loss + l
    cum_loss = cum_loss / nr_batches
    print('average val_loss: %f ' % cum_loss)

def predict():
    model.load_weights('weights/every_epoch.hdf5')
    batch_size_test = 50
    zip_path_test = '../data/test_jpg.zip'
    test_df = pd.read_csv('../data/nn_test.csv')
    seq_1 = data_sequence_test(test_df, batch_size=batch_size_test, zip_path=zip_path_test)
    preds = model.predict_generator(seq_1, workers=4,use_multiprocessing=True)
    np.save('../data/nn_predictions.npy', preds)
    submission = pd.read_csv('../data/nn_sample_submission.csv')
    submission['deal_probability'] = preds
    submission.to_csv("submission.csv", index=False)

predict()