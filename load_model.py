import pickle

def run_model():
    #Predict input data
    #Load model
    lasso_model = pickle.load(open("./save_model/lasso.sav", "rb"))
    ENet_model = pickle.load(open("./save_model/ENet.sav", "rb"))
    KRR_model = pickle.load(open("./save_model/KRR.sav", "rb"))
    GBoost_model = pickle.load(open("./save_model/GBoost.sav", "rb"))
    model_xgb_model = pickle.load(open("./save_model/model_xgb.sav", "rb"))
    model_lgb_model = pickle.load(open("./save_model/model_lgb.sav", "rb"))
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