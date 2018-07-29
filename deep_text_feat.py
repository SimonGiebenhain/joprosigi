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
from keras.initializers import he_uniform
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model
from sklearn.model_selection import KFold
from keras import layers

from gensim.models import word2vec

from keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


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
use_cols = ['item_id', 'param_1', 'param_2', 'param_3', 'title', 'description', 'deal_probability']

EMBEDDING_DIM1 = 300  # this is from the pretrained vectors


#TODO
#fill price with average of category
#treat image_top_1 as numerical, categorical or string?
#use activation weekday'''


def ztransform(col):
    mean = np.mean(col)
    std = np.std(col)
    return (col - mean) / std


def preprocess_dataset(dataset):


    dataset['description'].fillna('unknowntextfield', inplace=True)
    dataset['title'].fillna('unknowntextfield', inplace=True)
    dataset['param_1'].fillna('unknowntextfield', inplace=True)
    dataset['param_2'].fillna('unknowntextfield', inplace=True)
    dataset['param_3'].fillna('unknowntextfield', inplace=True)


    dataset['text'] = (dataset['title'] + "ß" + dataset['param_1'] + "ß" + dataset['param_2'] + "ß" + dataset['param_3'] + "ß" + dataset['description']).astype(str)
    del dataset['description'], dataset['title'], dataset['param_1'], dataset['param_2'], dataset['param_3']
    gc.collect()

    print("PreProcessing Function completed.")

    return dataset


def set_up_tokenizer(df):
    t = df['text'].values
    tokenizer = text.Tokenizer(num_words=500000)
    tokenizer.fit_on_texts(list(t))
    return df, tokenizer



def get_keras_data(dataset, tokenizer):
    t = tokenizer.texts_to_sequences(dataset['text'].astype('str'))
    X = {
        'seq_title_description': sequence.pad_sequences(t, maxlen=300),
    }
    return X

# TODO
# What about test data when calculating max value?
# Why +2??
# max_seq_title_description_length
# max_words_title_description


def prep_df():
    train = pd.read_csv("../data/train.csv", usecols = use_cols)
    test = pd.read_csv("../data/test.csv", usecols=use_cols[:-1])

    y_train = np.array(train['deal_probability'])
    del train['deal_probability']
    gc.collect()

    train_size = train.shape[0]
    all = pd.concat([train, test], axis=0)
    del train, test
    gc.collect()


    all = preprocess_dataset(all)
    all, tokenizer = set_up_tokenizer(all)
    print("Tokenization done and TRAIN READY FOR Validation splitting")

    with open('./others/own_tokenizer.pickle', 'wb') as handle:
       pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO
    # vocabsize + 2
    # dafuq is qoing on with c,c1,w_Y, w_No: WHY?

    EMBEDDING_FILE1 = '../data/embeddings/avito.w2v'
    max_num_words = 500000
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
    print('emb shape')
    print(embedding_matrix.shape)
    del embedding_matrix
    gc.collect()


    train = all[:train_size]
    train.to_csv('../data/new_rnn_train.csv')
    del train
    gc.collect()
    test = all[train_size:]
    test.to_csv('../data/new_rnn_test.csv')
    del test
    del all
    gc.collect()
    np.save('../data/new_rnn_y_train.npy', y_train)

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
    seq_title_description = Input(shape=[300], name="seq_title_description")

    # Embeddings layers
    embedding_matrix1 = np.load('./others/own_embedding_matrix_2.npy')
    vocab_size = embedding_matrix1.shape[0]
    print('Vocab size: % d' % vocab_size)
    emb_seq_title_description = Embedding(vocab_size, EMBEDDING_DIM1, weights=[embedding_matrix1], trainable=False)(
        seq_title_description)

    x = SpatialDropout1D(0.2)(emb_seq_title_description)
    x = Bidirectional(layers.CuDNNLSTM(128, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
    #x = add_common_layers(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256, kernel_initializer=he_uniform(seed=0))(x)
    x = layers.PReLU()(x)
    x = Dense(128, kernel_initializer=he_uniform(seed=0))(x)
    out = layers.PReLU()(x)
    out = BatchNormalization()(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=['seq_title_description'], outputs=out)
    model.compile(optimizer=Adam(lr=0.0005, clipnorm=0.5), loss='mean_squared_error', metrics=[root_mean_squared_error])
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
    train = pd.read_csv('../data/new_rnn_train.csv')
    y_train = np.load('../data/new_rnn_y_train.npy')
    train['deal_probability'] = y_train
    #train = train.sample(frac=0.1).reset_index(drop=True)
    y_train = train['deal_probability']
    del train['deal_probability']
    gc.collect()

    with open('./others/new_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Initialize a new Model for Current FOLD
    epochs = 5
    batch_size = 512 * 3
    #steps = (int(train.shape[0] / batch_size)) * epochs
    #lr_init, lr_fin = 0.009, 0.00045
    #lr_decay = exp_decay(lr_init, lr_fin, steps)
    modelRNN = RNN_model()
    #K.set_value(modelRNN.optimizer.lr, lr_init)
    #K.set_value(modelRNN.optimizer.decay, lr_decay)

    X_train = get_keras_data(train, tokenizer)

    # Fit the NN Model

    # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
    modelRNN.fit(X_train, y_train,
                 batch_size=batch_size, epochs=epochs,
                 validation_split=0.33,
                 shuffle=True,
                 verbose=1)
    # Predict test set for Kth Fold
    modelRNN.save_weights('./weights/new_rnn_with_add_feat.hdf5')

def gen_feat():
    train = pd.read_csv('../data/new_rnn_train.csv')
    y_train = np.load('../data/new_rnn_y_train.npy')
    train['deal_probability'] = y_train
    # train = train.sample(frac=0.1).reset_index(drop=True)
    y_train = train['deal_probability']
    del train['deal_probability']
    gc.collect()

    with open('./others/new_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Initialize a new Model for Current FOLD
    epochs = 5
    batch_size = 512 * 3
    # steps = (int(train.shape[0] / batch_size)) * epochs
    # lr_init, lr_fin = 0.009, 0.00045
    # lr_decay = exp_decay(lr_init, lr_fin, steps)
    modelRNN = RNN_model()
    # K.set_value(modelRNN.optimizer.lr, lr_init)
    # K.set_value(modelRNN.optimizer.decay, lr_decay)

    X_train = get_keras_data(train, tokenizer)

    # Fit the NN Model

    # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
    modelRNN.fit(X_train, y_train,
                 batch_size=batch_size, epochs=epochs,
                 validation_split=0.33,
                 shuffle=True,
                 verbose=1)
    # Predict test set for Kth Fold
    modelRNN.save_weights('./weights/new_rnn_with_add_feat.hdf5')


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

prep_df()
gc.collect()
train_only()