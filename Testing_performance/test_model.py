import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import numpy as np

import bentoml


def run_experiment(model, X_test, y_test):
#     model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_err = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test,y_pred)
    MSE = np.sqrt(mean_squared_error(y_test, y_pred))

    print("R^2 : ", r2_score(y_test, y_pred))
    print("MAE :", mean_absolute_error(y_test,y_pred))
    print("RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"r2_err":r2_err, "MAE":MAE, "MSE":MSE} 


def main():

    model = bentoml.sklearn.load_model("abalone_regressor_tree:latest")
    
    print(model)    

    full_columns_name = ['length', 'diameter', 'height', 'whole weight', 'shucked weight',
           'viscera weight', 'shell weight', 'rings', 'M', 'F', 'I']

    Test_dataset = pd.read_csv("testset.csv")


    df_X_testset = Test_dataset.drop(columns=['rings'])
    df_y_testset = Test_dataset.filter(['rings'])

    run_experiment(model, df_X_testset, df_y_testset)


if __name__ == '__main__':
    main()
