import pandas as pd
import numpy as np
import gc
from zipfile import ZipFile


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
    cols = ['matching_classes', 'xception_score', 'inception_score', 'inception_resnet_score', 'size',
            'avg_red', 'avg_green', 'avg_blue', 'std_red', 'std_green', 'std_blue', 'awp', 'blurriness']
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
    cols = cols + ['score_mean', 'score_std', 'avg_light_mean', 'avg_light_std', 'std_light_mean', 'std_light_std']
    for col in cols:
        base[col] = ztransform(base[col])
    base['xception'] = base['xception'].astype('category')
    base['inception'] = base['inception'].astype('category')
    base['inception_resnet'] = base['inception_resnet'].astype('category')
    return base


test = pd.read_csv('../data/test.csv')
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

#TODO: Fill -1 properly

test.replace([-1,xception_score_nematode, inception_score_nematode, inception_resnet_score_nematode], np.NaN, inplace=True)

test = extract_features(test)
print(test.isnull().sum().sort_values(ascending=False))

gc.collect()

train = pd.read_csv('../data/train.csv')
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

image_feature_train = pd.concat([ i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9], axis=0, ignore_index=True).reset_index(drop=True)

train_with_image = pd.concat([train_with_image, image_feature_train], axis=1)

no_image_index = np.logical_not(train['image'].isin(item_ids))
train_without_image = train.loc[no_image_index]
train = pd.concat([train_with_image, train_without_image], axis=0, ignore_index=True)
train = extract_features(train)
train = train.sample(frac=1).reset_index(drop=True)




print(train.isnull().sum().sort_values(ascending=False))

y_train = np.array(train['deal_probability'])
del train['deal_probability']
gc.collect()
train_size = train.shape[0]
all = pd.concat([train, test], axis=0)
del train, test
gc.collect()



