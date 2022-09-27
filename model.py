#Import libraries
import numpy as np
from process_data import pre_process
# from scipy import stats
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# from sklearn.model_selection import KFold, cross_val_score, train_test_split
# from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import pickle


def pre_train(dataset):
    df_train, df_test = pre_process(dataset)
    # print(df_test,df_train)

    if 'Unnamed: 0' in df_train.columns:
        df_train.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'Unnamed: 0' in df_test.columns:
        df_test.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    y_train = df_train['SalePrice']
    Y = np.exp(y_train)

    df_train.drop(['SalePrice'], axis=1, inplace=True)

    train = df_train
    return df_train, df_test, y_train
    


# def rmsle_cv(model):
#     n_folds = 5
#     kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(pre_train.train.values)
#     rmse = np.sqrt(-cross_val_score(model, pre_train.train.values, pre_train.y_train, scoring="neg_mean_squared_error", cv=kf))
#     return rmse

def models():
    # LASSO
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))
                        
    # Elastic Net Regression
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, random_state=42))

    # Kernel Ridge Regression
    KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)

    # Gradient Boosting Regression
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                    max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10, 
                                    loss='huber', random_state =42)

    # XGBoost
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                                learning_rate=0.05, max_depth=3, 
                                min_child_weight=1.7817, n_estimators=2200,
                                reg_alpha=0.4640, reg_lambda=0.8571,
                                subsample=0.5213, silent=1,
                                random_state =42, nthread = -1)

    # LightGBM
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                learning_rate=0.05, n_estimators=720,
                                max_bin = 55, bagging_fraction = 0.8,
                                bagging_freq = 5, feature_fraction = 0.2319,
                                feature_fraction_seed=9, bagging_seed=9,
                                min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

    return lasso,ENet,KRR,GBoost,model_xgb,model_lgb

def run_model(dataset):

    df_train,df_test,y_train = pre_train(dataset)
    lasso,ENet,KRR,GBoost,model_xgb,model_lgb = models()

    # x_train = df_train.to_numpy()
    # lasso.fit(x_train, y_train)
    # ENet.fit(x_train, y_train)
    # KRR.fit(x_train, y_train)
    # GBoost.fit(x_train, y_train)
    # model_xgb.fit(x_train, y_train)
    # model_lgb.fit(x_train, y_train)

    # #Save model
    # pickle.dump(lasso, open('./save_model/lasso.sav', 'wb'))
    # pickle.dump(ENet, open('./save_model/ENet.sav', 'wb'))
    # pickle.dump(KRR, open('./save_model/KRR.sav', 'wb'))
    # pickle.dump(GBoost, open('./save_model/GBoost.sav', 'wb'))
    # pickle.dump(model_xgb, open('./save_model/model_xgb.sav', 'wb'))
    # pickle.dump(model_lgb, open('./save_model/model_lgb.sav', 'wb'))

    #Load model
    lasso_model = pickle.load(open("./save_model/lasso.sav", "rb"))
    ENet_model = pickle.load(open("./save_model/ENet.sav", "rb"))
    KRR_model = pickle.load(open("./save_model/KRR.sav", "rb"))
    GBoost_model = pickle.load(open("./save_model/GBoost.sav", "rb"))
    model_xgb_model = pickle.load(open("./save_model/model_xgb.sav", "rb"))
    model_lgb_model = pickle.load(open("./save_model/model_lgb.sav", "rb"))
    
    #Predict input data
    x_test = df_test.to_numpy()
    lasso_predict = lasso_model.predict(x_test)
    ENet_predict = ENet_model.predict(x_test)
    KRR_predict = KRR_model.predict(x_test)
    GBoost_predict = GBoost_model.predict(x_test)
    XGB_predict = model_xgb_model.predict(x_test)
    LGB_predict = model_lgb_model.predict(x_test)

    y_predict = (lasso_predict + ENet_predict + KRR_predict + GBoost_predict + XGB_predict + LGB_predict)/6

    y_predict = np.exp(y_predict)
    y_predict = y_predict.astype(int)

    return y_predict[0]

#     """
#     Implement a house price prediction model

#     Args:
#         dataset (.csv): input data to predict house price

#     Returns:
#         outcome (float): house price after predicted
#     """
#     train_data = pd.read_csv("train2.csv")
#     test_data = pd.read_csv(dataset)
#     # test_data = pd.read_csv("train.csv")
#     pd.set_option("display.max_columns", None)  # View all columns
#     pd.set_option("display.max_rows", None)  # View all columns
#     # train_data.head(6)

#     feature = ['MSSubClass', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandSlope', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
#                'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
#                'BsmtUnfSF',
#                'TotalBsmtSF',
#                'HeatingQC',
#                'CentralAir',
#                '1stFlrSF',
#                '2ndFlrSF',
#                'LowQualFinSF',
#                'GrLivArea',
#                'BsmtFullBath',
#                'BsmtHalfBath',
#                'FullBath',
#                'HalfBath',
#                'BedroomAbvGr',
#                'KitchenAbvGr',
#                'KitchenQual',
#                'TotRmsAbvGrd',
#                'Functional',
#                'Fireplaces',
#                'FireplaceQu',
#                'GarageYrBlt',
#                'GarageFinish',
#                'GarageCars',
#                'GarageArea',
#                'GarageQual',
#                'GarageCond',
#                'PavedDrive',
#                'WoodDeckSF',
#                'OpenPorchSF',
#                'EnclosedPorch',
#                '3SsnPorch',
#                'ScreenPorch',
#                'PoolArea',
#                'MiscVal',
#                'MoSold',
#                'YrSold',
#                'TotalHouseSF',
#                'MSZoning_C (all)',
#                'MSZoning_FV',
#                'MSZoning_RH',
#                'MSZoning_RL',
#                'MSZoning_RM',
#                'LandContour_Bnk',
#                'LandContour_HLS',
#                'LandContour_Low',
#                'LandContour_Lvl',
#                'LotConfig_Corner',
#                'LotConfig_CulDSac',
#                'LotConfig_FR2',
#                'LotConfig_FR3',
#                'LotConfig_Inside',
#                'Neighborhood_Blmngtn',
#                'Neighborhood_Blueste',
#                'Neighborhood_BrDale',
#                'Neighborhood_BrkSide',
#                'Neighborhood_ClearCr',
#                'Neighborhood_CollgCr',
#                'Neighborhood_Crawfor',
#                'Neighborhood_Edwards',
#                'Neighborhood_Gilbert',
#                'Neighborhood_IDOTRR',
#                'Neighborhood_MeadowV',
#                'Neighborhood_Mitchel',
#                'Neighborhood_NAmes',
#                'Neighborhood_NPkVill',
#                'Neighborhood_NWAmes',
#                'Neighborhood_NoRidge',
#                'Neighborhood_NridgHt',
#                'Neighborhood_OldTown',
#                'Neighborhood_SWISU',
#                'Neighborhood_Sawyer',
#                'Neighborhood_SawyerW',
#                'Neighborhood_Somerst',
#                'Neighborhood_StoneBr',
#                'Neighborhood_Timber',
#                'Neighborhood_Veenker',
#                'Condition1_Artery',
#                'Condition1_Feedr',
#                'Condition1_Norm',
#                'Condition1_PosA',
#                'Condition1_PosN',
#                'Condition1_RRAe',
#                'Condition1_RRAn',
#                'Condition1_RRNe',
#                'Condition1_RRNn',
#                'Condition2_Artery',
#                'Condition2_Feedr',
#                'Condition2_Norm',
#                'Condition2_PosA',
#                'Condition2_PosN',
#                'Condition2_RRAe',
#                'Condition2_RRAn',
#                'Condition2_RRNn',
#                'BldgType_1Fam',
#                'BldgType_2fmCon',
#                'BldgType_Duplex',
#                'BldgType_Twnhs',
#                'BldgType_TwnhsE',
#                'HouseStyle_1.5Fin',
#                'HouseStyle_1.5Unf',
#                'HouseStyle_1Story',
#                'HouseStyle_2.5Fin',
#                'HouseStyle_2.5Unf',
#                'HouseStyle_2Story',
#                'HouseStyle_SFoyer',
#                'HouseStyle_SLvl',
#                'RoofStyle_Flat',
#                'RoofStyle_Gable',
#                'RoofStyle_Gambrel',
#                'RoofStyle_Hip',
#                'RoofStyle_Mansard',
#                'RoofStyle_Shed',
#                'RoofMatl_ClyTile',
#                'RoofMatl_CompShg',
#                'RoofMatl_Membran',
#                'RoofMatl_Metal',
#                'RoofMatl_Roll',
#                'RoofMatl_Tar&Grv',
#                'RoofMatl_WdShake',
#                'RoofMatl_WdShngl',
#                'Exterior1st_AsbShng',
#                'Exterior1st_AsphShn',
#                'Exterior1st_BrkComm',
#                'Exterior1st_BrkFace',
#                'Exterior1st_CBlock',
#                'Exterior1st_CemntBd',
#                'Exterior1st_HdBoard',
#                'Exterior1st_ImStucc',
#                'Exterior1st_MetalSd',
#                'Exterior1st_Plywood',
#                'Exterior1st_Stone',
#                'Exterior1st_Stucco',
#                'Exterior1st_VinylSd',
#                'Exterior1st_Wd Sdng',
#                'Exterior1st_WdShing',
#                'Exterior2nd_AsbShng',
#                'Exterior2nd_AsphShn',
#                'Exterior2nd_Brk Cmn',
#                'Exterior2nd_BrkFace',
#                'Exterior2nd_CBlock',
#                'Exterior2nd_CmentBd',
#                'Exterior2nd_HdBoard',
#                'Exterior2nd_ImStucc',
#                'Exterior2nd_MetalSd',
#                'Exterior2nd_Other',
#                'Exterior2nd_Plywood',
#                'Exterior2nd_Stone',
#                'Exterior2nd_Stucco',
#                'Exterior2nd_VinylSd',
#                'Exterior2nd_Wd Sdng',
#                'Exterior2nd_Wd Shng',
#                'MasVnrType_BrkCmn',
#                'MasVnrType_BrkFace',
#                'MasVnrType_None',
#                'MasVnrType_Stone',
#                'Foundation_BrkTil',
#                'Foundation_CBlock',
#                'Foundation_PConc',
#                'Foundation_Slab',
#                'Foundation_Stone',
#                'Foundation_Wood',
#                'Heating_Floor',
#                'Heating_GasA',
#                'Heating_GasW',
#                'Heating_Grav',
#                'Heating_OthW',
#                'Heating_Wall',
#                'Electrical_FuseA',
#                'Electrical_FuseF',
#                'Electrical_FuseP',
#                'Electrical_Mix',
#                'Electrical_SBrkr',
#                'GarageType_2Types',
#                'GarageType_Attchd',
#                'GarageType_Basment',
#                'GarageType_BuiltIn',
#                'GarageType_CarPort',
#                'GarageType_Detchd',
#                'GarageType_None',
#                'SaleType_COD',
#                'SaleType_CWD',
#                'SaleType_Con',
#                'SaleType_ConLD',
#                'SaleType_ConLI',
#                'SaleType_ConLw',
#                'SaleType_New',
#                'SaleType_Oth',
#                'SaleType_WD',
#                'SaleCondition_Abnorml','SaleCondition_AdjLand','SaleCondition_Alloca','SaleCondition_Family','SaleCondition_Normal','SaleCondition_Partial']

#     X = train_data[feature]
#     y = train_data.SalePrice
#     test = test_data[feature]

#     clf = RandomForestRegressor()
#     clf.fit(X, y)

#     outcome = clf.predict(test)
#     # print(type(outcome[0]))
#     return outcome[0]
