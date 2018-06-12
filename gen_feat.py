from zipfile import ZipFile
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import gc
import cv2
import keras

from keras.models import Model
from keras.utils import Sequence
from keras import Input
from keras.preprocessing import image
import keras.applications.xception as xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.layers import Lambda, Concatenate
from keras.layers import Dense
from tensorflow.python.keras.initializers import Identity
from matplotlib import pyplot as plt




class data_sequence(Sequence):
    def __init__(self, x_set, batch_size, zip_path):
        self.x = x_set
        self.batch_size = batch_size
        self.zip_path = zip_path

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        image_hashes = batch_x.image.values
        images = np.zeros((self.batch_size, 299, 299, 3), dtype=np.float32)
        size = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        avg_red = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        avg_green = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        avg_blue = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        std_red = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        std_green = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)
        std_blue = -1 * np.ones(shape=(self.batch_size,1), dtype=np.float32)


        with ZipFile(self.zip_path) as im_zip:
            for i,hash in enumerate(image_hashes):
                try:
                    stats = im_zip.getinfo(hash + '.jpg')
                    size[i] = stats.file_size
                    file = im_zip.open(hash + '.jpg')

                    img = image.load_img(file, target_size=(299,299))
                    img = image.img_to_array(img)

                    average_color = [img[:, :, i].mean() for i in range(3)]
                    std_color = [img[:, :, i].std() for i in range(3)]
                    avg_red[i] = average_color[0]
                    avg_green[i] = average_color[1]
                    avg_blue[i] = average_color[2]
                    std_red[i] = std_color[0]
                    std_green[i] = std_color[1]
                    std_blue[i] = std_color[2]
                    #im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #blurriness = cv2.Laplacian(im, cv2.CV_32F).var()

                    images[i] = xception.preprocess_input(img)
                except KeyError:
                    print('Error loading image: %d' % hash)
        return {'inp':images, 'feat_in': np.concatenate([size, avg_red, avg_green, avg_blue, std_red, std_green, std_blue], axis=1)}



train_df = pd.read_csv('../data/train.csv')
zip_path = '../data/train_jpg_0.zip'
files_in_zip = ZipFile(zip_path).namelist()
item_ids = list(map(lambda s: os.path.splitext(s)[0], files_in_zip))
train_df = train_df.loc[train_df['image'].isin(item_ids)]
#for i in range(10):
#    hash = train_df.image.values
#    with ZipFile(zip_path) as im_zip:
#        file = im_zip.open(hash[i] + '.jpg')
        #img = image.img_to_array(img)
#        img = image.load_img(file, target_size=(299,299))
#        plt.subplot(1,10,i+1)
#        plt.imshow(img)
        #plt.subplot(1,2,2)
        #plt.imshow(small_img)
#plt.show()

xception_model = xception.Xception(weights='imagenet')
inception_model = InceptionV3(weights='imagenet')
inception_resnet_model = InceptionResNetV2(weights='imagenet')

input = Input(shape=[299,299,3], name='inp')
feat_in = Input(shape=[7], dtype=np.float32, name='feat_in')

x = xception_model(input)
y = inception_model(input)
z = inception_resnet_model(input)

feat_out = Dense(7, use_bias=False, kernel_initializer=keras.initializers.Identity(), name='Identity', trainable=False)(feat_in)

model = Model([input, feat_in],
              [x, y, z, feat_out])
batch_size = 100
data_gen = data_sequence(x_set=train_df, batch_size=batch_size, zip_path=zip_path)
preds = model.predict_generator(data_gen,
                        max_queue_size=8, workers=4, use_multiprocessing=True,
                        verbose=1)
im_feat = preds[-1]
preds = preds[:-1]
im_features = pd.DataFrame(im_feat, columns=['size', 'avg_red', 'avg_green', 'avg_blue',
                                             'std_red', 'std_green', 'std_blue'])
print(im_features.head())

#TODO Merge im_features, then concat predictions
preds = [xception.decode_predictions(preds[0], top=1), xception.decode_predictions(preds[1], top=1),
          xception.decode_predictions(preds[2], top=1)]
print(preds)
preds = np.squeeze(np.stack(preds, axis=0))

np.save('../data/train_0_preds.npy', preds)
im_features.to_csv('../data/im_feat_train_0.csv')

print(preds.shape)
im_predictions_1 = pd.DataFrame(np.transpose(preds[:,:,1]), columns=['xception', 'inception', 'inception_resnet'])
im_predictions_2 = pd.DataFrame(np.transpose(preds[:,:,2]), columns=['xception_score', 'inception_score', 'inception_resnet_score'])
im_preds = pd.concat([im_predictions_1, im_predictions_2, im_features], axis=1)
im_preds.to_csv('../data/im_preds_299.csv')

#im_features = pd.concat([im_features, im_predictions_1, im_predictions_2], axis=1)
#print(im_features[:10])