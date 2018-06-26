import pandas as pd
import numpy as np
import gc

use_cols_active = ['item_id', 'category_name', 'param_1', 'param_2', 'param_3']

train_active = pd.read_csv('../data/train_active.csv')
test_active = pd.read_csv('../data/test_active.csv')
train_periods = pd.read_csv('../data/periods_train.csv')
test_periods = pd.read_csv('../data/periods_test.csv')

train_active = train_active.merge(train_periods, on='item_id', how='left')
test_active = test_active.merge(test_periods, on='item_id', how='left')

del train_periods, test_periods
gc.collect()

all = pd.concat([train_active, test_active]).drop_duplicates().reset_index(drop=True)
all['days_up'] = all['date_to'].dt.dayofyear - all['date_from'].dt.dayofyear

gp = all.groupby(['item_id'])[['days_up']]

gp_df = pd.DataFrame()
gp_df['days_up_sum'] = gp.sum()['days_up']
gp_df['times_put_up'] = gp.count()['days_up']
gp_df.reset_index(inplace=True)
gp_df.rename(index=str, columns={'index': 'item_id'})

print(gp_df.head())

all.drop_duplicates(['item_id'], inplace=True)
all = all.merge(gp_df, on='item_id', how='left')
print(all.head())

all['full_cat'] = all['category_name'] + ' ' + all['param_1'] + ' ' + all['param_2'] + ' ' + all['param_3']


df = all.groupby('full_cat', as_index=False)['days_up_sum', 'times_put_up'].mean().reset_index().rename(index=str, columns={
        'days_up_sum': 'avg_days_up_cat',
        'times_put_up': 'avg_times_up_cat'
    })

nb_ads_per_category = all.groupby('full_cat', as_Index=False)['days_up'].count().reset_index().rename(index=str, columns={
        'days_up': 'n_cat_items'
})

all = all.merge(df, on=['full_cat'], how='left')
all = all.merge(nb_ads_per_category, on='category_name', how='left')

print(all.head())

all = all['category_name', 'avg_days_up_cat', 'avg_times_up_cat', 'n_cat_items']

print(all.head())

all.to_csv('../data/aggregated_cat_feat.csv')

