import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb


raw_train = pd.read_csv('./data/cleaned_train.csv')
# raw_train = pd.read_csv('./data/train.csv')
raw_test = pd.read_csv('./data/test.csv')
submit_df = pd.read_csv('./data/submit_example.csv')

for df in [raw_train, raw_test]:
    # 处理空值
    for f in ['category_code', 'brand']:
        df[f].fillna('<unkown>', inplace=True)

    # 处理时间
    df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S UTC')
    df['timestamp'] = df['event_time'].apply(lambda x: time.mktime(x.timetuple()))
    df['timestamp'] = df['timestamp'].astype(int)

# 排序
raw_train = raw_train.sort_values(['user_id', 'timestamp'])
raw_test = raw_test.sort_values(['user_id', 'timestamp'])

df = pd.concat([raw_train, raw_test], ignore_index=True)

for f in ['category_code', 'brand']:
    # 构建编码器
    le = LabelEncoder()
    le.fit(df[f])

    # 设置新值
    raw_train[f] = le.transform(raw_train[f])
    raw_test[f] = le.transform(raw_test[f])

le = LabelEncoder()
le.fit(df['event_type'])
raw_train['event_type'] = le.transform(raw_train['event_type'])
raw_test['event_type'] = le.transform(raw_test['event_type'])
    
# 删除无用列
useless = ['event_time', 'user_session', 'timestamp']
for df in [raw_train, raw_test]:
    df.drop(columns=useless, inplace=True)

# 训练集数据生成：滑动窗口
# 用前一个时间节点的数据预测后一个时间节点是商品
train_df = pd.DataFrame()
user_ids = raw_train['user_id'].unique()

# slide_len 窗口长度
slide_len = 10
for uid in tqdm(user_ids):
    user_data = raw_train[raw_train['user_id'] == uid].copy(deep=True)
    # if user_data.shape[0] < 2:
    if user_data.shape[0] < slide_len + 1 or user_data.shape[0] > 300:   # 0.12544803000  -> 0.12724014000 有用
        # 小于两条或者大于300条的，直接忽略
        continue
    for i in range(1,slide_len):
        name = 'product'+str(i)
        event = 'event'+str(i)
        user_data[name] = user_data['product_id'].shift(-i)
        user_data[event] = user_data['event_type'].shift(-i)
    user_data['y'] = user_data['product_id'].shift(-slide_len)
    # now_purchase = user_data['product_id'].iloc[-1]
    # for index, row in reversed(list(user_data.iterrows())):
    #     user_data.loc[index,'y'] = now_purchase
    #     # print(index)
    #     if row['event_type']=='purchase':
    #         # print('now_purchase',row['product_id'])
    #         now_purchase = row['product_id']
    user_data = user_data.head(user_data.shape[0]-slide_len)
    # print(uid)
    # print(user_data)
    train_df = pd.concat([train_df, user_data])
    # exit()

train_df['y'] = train_df['y'].astype(int)
for i in range(1,slide_len):
    name = 'product'+str(i)
    event = 'event'+str(i)
    train_df[name] = train_df[name].astype(int)
    train_df[event] = train_df[event].astype(int)
train_df = train_df.reset_index(drop=True)
train_df.to_csv('./data/slide_window.csv', index=False)
# event_type编码
# le = LabelEncoder()
# le.fit(df['event_type'])
# train_df['event_type'] = le.transform(train_df['event_type'])
# raw_test['event_type'] = le.transform(raw_test['event_type'])
train_df.drop(columns=['user_id'], inplace=True)

# 测试集数据生成，只取每个用户最后一次操作用来做预测
test_df = raw_test.groupby(['user_id'], as_index=False).last()

user_ids = test_df['user_id'].unique()

preds = []
count = 0
for uid in tqdm(user_ids):
    pids = raw_test[raw_test['user_id'] == uid]['product_id'].unique()

    # 找到训练集中有这些product_id的数据作为当前用户的训练集
    p_train = train_df[train_df['product_id'].isin(pids)]
    # p_train = train_df
    
    # 对test中每个uid对应的记录取最后五条
    user_test_data = raw_test[raw_test['user_id'] == uid].copy(deep=True)
    user_test_data.drop(columns='user_id',inplace=True)
    if user_test_data.shape[0] < slide_len + 1:
        # 小于slide_len+1 
        # pred = user_test_data['product_id'].iloc[-1]
        pred = 0
        preds.append(pred)
        count+=1
        continue

    for i in range(1,slide_len):
        name = 'product'+str(i)
        event = 'event'+str(i)
        user_test_data[name] = user_test_data['product_id'].shift(-i)
        user_test_data[event] = user_test_data['event_type'].shift(-i)
    
    user_test = user_test_data.iloc[-slide_len]
    for i in range(1,slide_len):
        name = 'product'+str(i)
        event = 'event'+str(i)
        user_test[name] = user_test[name].astype(int)
        user_test[event] = user_test[event].astype(int)
    
    # 只取最后一条进行预测
    # user_test = test_df[test_df['user_id'] == uid].drop(columns=['user_id'])

    X_train = p_train.iloc[:, :-1]
    y_train = p_train['y']
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    if len(X_train) > 0:
        # 训练
        clf = lgb.LGBMClassifier(**{'seed': int(time.time())})
        # clf = lgb.LGBMClassifier(**{'seed': int(time.time(),boosting_type='rf')})
        # clf.fit(X_train, y_train,eval_set=[(X_val, y_val)])
        clf.fit(X_train, y_train)
        
        # test = user_test.drop('user_id', axis=1)
        # print(test)
        # print(test.shape)
        # exit()
        # 预测
        pred = clf.predict([user_test])[0]
        # print(user_test)
    else:
        # 训练集中无对应数据
        # 直接取最后一条数据作为预测值
        pred = user_test['product_id'].iloc[0]

    preds.append(pred)
print('count:',count)

submit_df['product_id'] = preds

# 分数 0.206
submit_df.to_csv('baseline.csv', index=False)



