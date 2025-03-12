import pandas as pd
df_train = pd.read_feather('data/ETHUSDT/df_train.feather')
df_val = pd.read_feather('data/ETHUSDT/df_val.feather')
df_test = pd.read_feather('data/ETHUSDT/df_test.feather')

df_train= df_train.head(20000)
df_val=df_val.head(20000)
df_test=df_test.head(20000)

df_train.to_feather('data/ETHUSDT/df_train.feather')
df_val.to_feather('data/ETHUSDT/df_val.feather')
df_test.to_feather('data/ETHUSDT/df_test.feather')

df_test = pd.read_feather('data/ETHUSDT/df_test.feather')

print(df_test.shape)