import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
#%%
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
#%%
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
#%%
train = pd.read_csv('files-pik_digital_day/train.csv', parse_dates=['date1'])
#%%
df_flat = pd.read_csv('files-pik_digital_day/flat.csv', parse_dates=True, encoding = "utf-8")
#%%
bulk_id = train['bulk_id'].unique().tolist()
#%%
train_grouped = train[['bulk_id', 'spalen', 'date1', 'price', 'value']].\
                    groupby(['bulk_id', 'spalen'])
#%%
train_gr_agg   = train_grouped.agg({'date1': np.min,
                    'price': np.mean,
                    'value': np.mean})
#%%
train_gr_agg = train_gr_agg.reset_index().drop(['spalen', 'price', 'value'], axis = 1).\
                drop_duplicates('bulk_id')
#%%
#%%
train['m_from_sale'] = 0
#%%

#%%
f, ax = plt.subplots(nrows = 2, ncols =1, sharex=True, figsize=(15, 12))
ax[0].plot(train[(train['bulk_id'] == bulk_id[2]) & (train['spalen'] == 1)]['date1'],
        train[(train['bulk_id'] == bulk_id[2]) & (train['spalen'] == 1)]['price'])
ax[1].plot(train[(train['bulk_id'] == bulk_id[2]) & (train['spalen'] == 1)]['date1'],
        train[(train['bulk_id'] == bulk_id[2]) & (train['spalen'] == 1)]['value'].cumsum())
ax[1].plot(train[(train['bulk_id'] == bulk_id[2]) & (train['spalen'] == 1)]['date1'],
        train[(train['bulk_id'] == bulk_id[2]) & (train['spalen'] == 1)]['start_square'])
#%%

ind = 10
print(train[(train['bulk_id'] == bulk_id[ind]) & (train['spalen'] == 1)]['date1'].shape)
print(len(train[(train['bulk_id'] == bulk_id[ind]) & (train['spalen'] == 1)]['date1'].unique()))
#%%
train[(train['bulk_id'] == bulk_id[1]) & (train['spalen'] == 1)]
#%%
#plt.scatter(train[['price', 'mean_sq']].values)
sns.regplot('spalen', 'price',train)
#%%
test = pd.read_csv('files-pik_digital_day/test.csv')
#%%
#train_filt = train[train['value'] != 0].copy()
train_filt = train.copy()
#%%
feat_to_del = 'start_square \
        plan_s\
        plan_m\
        plan_l\
        vid_0\
        vid_1\
        vid_2'
feat_to_del = feat_to_del.split()
feat_to_del += ['date1', 'id', 'Лифт']
#%%

#%%
train_filt = train_filt.drop(feat_to_del, axis = 1)
#%%
cat_features = train_filt.columns[train_filt.dtypes == 'object'].tolist() + \
                ['spalen', 'Кондиционирование', 'Вентлияция', 'Видеонаблюдение']
#%%

#%%
non_obj_features = sorted(list(set(train_filt.columns.tolist())
                                - set(cat_features)))
#%%
cat_features
#%%
sns.pairplot(train_filt[non_obj_features[18:24]])
#%%
v = LabelEncoder()
#%%
for feat in cat_features:
    train_filt[feat] = v.fit_transform(train_filt[feat])

#%%
print(train_filt.shape)
#%%
sns.distplot(train_filt['value'], fit=stats.norm)
#%%
(mu, sigma) = stats.norm.fit(train_filt['value'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#%%
stats.probplot(train_filt['value'], plot=plt)
#%%
train_filt['value'] = np.log1p(train_filt['value'])
#%%
sns.distplot(train_filt['value'], fit=stats.norm)
#%%
(mu, sigma) = stats.norm.fit(train_filt['value'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#%%
stats.probplot(train_filt['value'], plot=plt)
#%%
skewed_feats = train_filt[non_obj_features].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
#%%
train_filt[cat_features] = train_filt[cat_features].astype(str)
#%%
train_filt = pd.get_dummies(train_filt)
print(train_filt.shape)
#%%
#%%
X_train = train_filt.drop('value',axis = 1).values
y_train = train_filt['value']
#%%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)
#%%
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    mse= np.sqrt(-cross_val_score(model, X_train, y_train, 
                                    scoring="neg_mean_squared_error", cv = kf))
    return(mse)
#%%
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#%%
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
#%%
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
#%%
model_lgb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
#%%
model_lgb.fit(X_train, y_train)
#%%
y_pred_train = model_lgb.predict(X_train)
y_pred_val = model_lgb.predict(X_val)
#%%
y_pred_train_in = np.exp(y_pred_train) - 1
y_pred_val_in = np.exp(y_pred_val) - 1
y_train_in = np.exp(y_train) - 1
y_val_in = np.exp(y_val) - 1
#%%
np.sqrt(mse(y_train_in, y_pred_train_in))
#%%
np.sqrt(mse(y_val_in, y_pred_val_in))
