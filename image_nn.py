from keras import layers
from keras import models
import pandas as pd
from zipfile import ZipFile
import os
import keras.backend as K
import gc
from keras.utils import Sequence
import numpy as np
from keras.preprocessing import image
from keras.applications import xception
from keras import optimizers
from keras import activations
from keras.models import Model

img_height = 224
img_width = 224
img_channels = 3


def add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    return y

class data_sequence(Sequence):
    def __init__(self, x_set, batch_size, zip_path, size):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path
        self.size = size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        image_hashs = batch_x.image.values
        images = np.zeros((self.batch_size, self.size, self.size, img_channels), dtype=np.float32)
        with ZipFile(self.zip_path) as im_zip:
            for i,hash in enumerate(image_hashs):
                try:
                    file = im_zip.open(hash + '.jpg')
                    img = image.load_img(file, target_size=(self.size, self.size))
                    img = image.img_to_array(img)
                    if not(np.isinf(img).any() or np.isnan(img).any()):
                        img = xception.preprocess_input(img)
                    images[i] = img
                except KeyError:
                    print('Error loading image %s' % hash)
                except IOError:
                    print('Error loading image %s' % hash)
        X = {'input_1': images}
        if (idx + 1) * self.batch_size > len(self.x.index):
           Y = np.ones((self.batch_size)) * 0.14
           Y[0: len(batch_x.index)] = batch_x.deal_probability.values
        else:
            Y = batch_x.deal_probability.values
        return X, Y


    def set_x_set(self, x_set):
        self.x = x_set

class data_sequence_pred(Sequence):
    def __init__(self, x_set, batch_size, zip_path, size):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path
        self.size = size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        image_hashs = batch_x.image.values
        images = np.zeros((self.batch_size, self.size, self.size, img_channels), dtype=np.float32)
        with ZipFile(self.zip_path) as im_zip:
            for i,hash in enumerate(image_hashs):
                try:
                    file = im_zip.open(hash + '.jpg')
                    img = image.load_img(file, target_size=(self.size, self.size))
                    img = image.img_to_array(img)
                    if not(np.isinf(img).any() or np.isnan(img).any()):
                        img = xception.preprocess_input(img)
                    images[i] = img
                except KeyError:
                    print('Error loading image %s' % hash)
                except IOError:
                    print('Error loading image %s' % hash)
        X = {'input_1': images}
        return X


    def set_x_set(self, x_set):
        self.x = x_set

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
    x = layers.Dense(1, activation=activations.sigmoid)(x)

    return x

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def train_only():
    train = pd.read_csv('../data/train.csv')

    zip_path = '../data/train_jpg.zip'
    files_in_zip = ZipFile(zip_path).namelist()
    item_ids = list(map(lambda s: os.path.splitext(s)[0], files_in_zip))
    train = train.loc[train['image'].isin(item_ids)]
    train_len = int(len(train.index) * 0.33)
    val = train[-train_len:]
    train = train[:-train_len]



    batch_size = 64
    lr = 0.005
    train = train.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)


    for i in range(2):
        if i == 0:
            data_seq_train = data_sequence(train, batch_size, zip_path, img_width)
            data_seq_val = data_sequence(val, batch_size, zip_path, img_width)
            image_tensor = layers.Input(shape=(img_height, img_width, img_channels), name='input_1')
            x = residual_network(image_tensor)
            model = models.Model(inputs=[image_tensor], outputs=[x])
            model.compile(optimizer=optimizers.Adam(lr=lr), loss=root_mean_squared_error,
                          metrics=[root_mean_squared_error])
            print(model.summary())
        else:
            data_seq_train = data_sequence(train, batch_size, zip_path, 299)
            data_seq_val = data_sequence(val, batch_size, zip_path, 299)
            #image_tensor = layers.Input(shape=(299, 299, img_channels), name='input_1')
            base_model = xception.Xception(weights='imagenet', include_top=False)
            for layer in base_model.layers[:125]:
                layer.trainable = False
            for layer in base_model.layers[125:]:
                layer.trainable = True
            x_base = base_model.output
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x_base)
            x = layers.Dense(512)(layers.Dropout(0.5)(x))
            x = add_common_layers(x)
            x = layers.Dense(64)(layers.Dropout(0.5)(x))
            x = add_common_layers(x)
            x = layers.Dense(1, activation='sigmoid')(x)
            model = Model([base_model.input], x)
            model.compile(optimizer=optimizers.Adam(lr=lr),
                          loss=root_mean_squared_error,
                          metrics=[root_mean_squared_error])
            model.summary()


        past_best = 0
        # TODO: sample training data, such that there are smaller epochs, maybe with callbackk
        loss = 0
        it = 50
        pat = 0
        for i in range(it):
            print('Iteration %d begins' % i)
            model.fit_generator(data_seq_train, steps_per_epoch=200, epochs=1,
                               verbose=0,
                                max_queue_size=16, workers=8, use_multiprocessing=True,)
            hist = model.evaluate_generator(data_seq_val, steps=100,
                                            max_queue_size=8, workers=8, use_multiprocessing=True)
            train = train.sample(frac=1).reset_index(drop=True)
            data_seq_train.set_x_set(train)
            val = val.sample(frac=1).reset_index(drop=True)
            data_seq_val.set_x_set(val)
            if i > 0:
                curr_loss = hist.history['val_loss'][-1]
                if  curr_loss < past_best:
                    model.save_weights('./weights/image_nn_serve_%d.hdf5' % i)
                    print('Loss improved from %f to %f!' % (past_best, curr_loss))
                    past_best = hist.history['val_loss'][-1]
                else:
                    pat += 1
                    print('Loss did not improve from %f!' % past_best)
                    if pat == 2:
                        lr /= 2
                        K.set_value(model.optimizer.lr, lr)
                        pat = 0

            else:
                past_best = hist.history['val_loss'][-1]
            print(hist.history['val_loss'])
            loss += hist.history['val_loss'][-1]
            gc.collect()
        print('avg_loss for model %d: %f' % (i,loss/it))

def predict():
    test = pd.read_csv('../data/train.csv')
    print(test.head())

    zip_path = '../data/train_jpg_0.zip'
    files_in_zip = ZipFile(zip_path).namelist()
    item_ids = list(map(lambda s: os.path.splitext(s)[0], files_in_zip))
    print(item_ids[0:5])
    test = test.loc[test['image'].isin(item_ids)]
    print(len(test.index))

    image_tensor = layers.Input(shape=(img_height, img_width, img_channels), name='input_1')
    network_output = residual_network(image_tensor)
    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    #model.compile(optimizer=optimizers.Adam(lr=0.005), loss=root_mean_squared_error, metrics=[root_mean_squared_error])
    #print(model.summary())

    model.load_weights('./weights/image_nn.hdf5', by_name=True, skip_mismatch=True)
    batch_size = 32

    data_seq_test = data_sequence_pred(test, batch_size, zip_path)

    preds = model.predict_generator(data_seq_test, steps=20, workers=4, use_multiprocessing=True)
    print(preds)

def evaluate():
    val = pd.read_csv('../data/train.csv')
    zip_path = '../data/train_jpg_0.zip'
    files_in_zip = ZipFile(zip_path).namelist()
    item_ids = list(map(lambda s: os.path.splitext(s)[0], files_in_zip))
    val = val.loc[val['image'].isin(item_ids)]
    val = val.sample(frac=1).reset_index(drop=True)

    image_tensor = layers.Input(shape=(img_height, img_width, img_channels), name='input_1')
    network_output = residual_network(image_tensor)
    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    model.compile(optimizer=optimizers.Adam(lr=0.005), loss=root_mean_squared_error, metrics=[root_mean_squared_error])
    # print(model.summary())

    model.load_weights('./weights/image_nn.hdf5', by_name=True, skip_mismatch=True)
    batch_size = 32

    data_seq_val = data_sequence(val, batch_size, zip_path)

    preds = model.evaluate_generator(data_seq_val, steps=20, workers=4, use_multiprocessing=True)
    print(preds)


train_only()