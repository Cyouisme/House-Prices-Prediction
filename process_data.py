import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

from scipy import stats
from scipy.stats import norm, skew
# from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p

def pre_process(dataset):
    train_data_path = r"D:\AI-ML\AICamp\House_Price_Prediction\data\train.csv"
    # test_data_path = pd.read_csv(dataset)

    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(dataset)

    train_ID = df_train['Id']
    test_ID = df_test['Id']

    df_train.drop("Id", axis=1, inplace=True)
    df_test.drop("Id", axis=1, inplace=True)

    y_train = df_train['SalePrice']

    ntrain = df_train.shape[0]
    ntest = df_test.shape[0]
    y_train = df_train.SalePrice.values
    data = pd.concat((df_train, df_test)).reset_index(drop=True)
    data.drop(['SalePrice'], axis=1, inplace=True)

    # Check missing values
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])

    # Drop feature has high rate missing values (>80%)
    data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)
    data.drop(['Utilities'], axis=1, inplace=True)

    data['YearBuilt'] = data['YearBuilt'].fillna(df_train['YearBuilt'].median())
    data['YrSold'] = data['YrSold'].fillna(df_train['YrSold'].median())
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']

    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)

    num_col = data.select_dtypes(exclude='object').columns
    cat_col = data.select_dtypes(include='object').columns

    df_train = data[:ntrain]

    # Fill NaN in num_col by median
    for col in num_col:
        data[col] = data[col].fillna(df_train[col].median())

    #Outliers
    num_data = data[num_col]
    cat_data = data[cat_col]

    num_data_train = num_data[:ntrain]
    cat_data_train = cat_data[:ntrain]
    print(num_data_train.shape)
    print(cat_data_train.shape)

    # Calculate the z-scores
    z_scores = stats.zscore(num_data_train)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 4).all(axis=1)
    num_data_train = num_data_train[filtered_entries]
    cat_data_train = cat_data_train[filtered_entries]

    y_train = y_train[filtered_entries]
    y_train.shape

    ntrain = num_data_train.shape[0]
    train_data = pd.concat([num_data_train, cat_data_train], axis=1, join='inner')

    test_data = data[1460:]

    data = pd.concat((train_data, test_data)).reset_index(drop=True)

    # Fill NaN in cat_col by None or Mode
    # None: FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond,
    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2,MasVnrType
    # Mode: SaleType,Exterior1st, Exterior2nd, KitchenQual, Electrical,Functional, MSZoning
    fill_none = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType']
    fill_mode = ['SaleType','Exterior1st', 'Exterior2nd', 'KitchenQual', 'Electrical','Functional', 'MSZoning']

    for col in fill_none:
        data[col] = data[col].fillna('None')
    for col in fill_mode:
        data[col] = data[col].fillna(df_train[col].mode()[0])

        
    data.isnull().sum().max()

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    for c in cols:
        if c not in data.columns: continue
        print(c)
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))

    # Adding total sqfootage feature
    data['TotalHouseSF'] = data['1stFlrSF'] + data['2ndFlrSF']

    numeric_features = data.dtypes[data.dtypes != "object"].index

    skewed_features = data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' : skewed_features})

    skewness = skewness[abs(skewness)>0.75]

    skewed_features = skewness.index
    lamb = 0.15
    for feature in skewed_features:
        data[feature] = boxcox1p(data[feature], lamb)

    data = pd.get_dummies(data)

    train = data[:ntrain]
    test = data[ntrain:]

    train['SalePrice'] = y_train

    # train.to_csv('train3v1.csv')
    # test.to_csv('test3v1.csv')
    return train, test