import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from math import sqrt

running_data = pd.read_csv('raw-data-kaggle.csv', sep=';')

#Target
Y = running_data.ElapsedTime
#Features
running_data_features = ['Distance','ElevationGain', 'AverageHeartRate'] 
#not including gender, missing values not imputable
#both distance and elevation gain have a pearsonr of about .20
X = running_data[running_data_features]

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state=1, test_size=.2)

cols_missing_vals = [col for col in running_data_features if train_X[col].isnull().any()]
if len(cols_missing_vals) > 0:
    print(f'Missing values in: {cols_missing_vals}. Imputing...') #missing values for heart rate
    myimputer = SimpleImputer() #fill in with avg heart rate
    imputed_train_X = pd.DataFrame(myimputer.fit_transform(train_X))
    imputed_valid_X = pd.DataFrame(myimputer.transform(val_X))
    imputed_train_X.columns = train_X.columns
    imputed_valid_X.columns = val_X.columns
    val_X = imputed_valid_X
    train_X = imputed_train_X


#XGBRegressor(n_estimators = 100, seed=2369, learning_rate=0.05, n_jobs=4, random_state=0)  RMSE: 11374.832395661719
#LinearRegression(n_jobs=4) RMSE: 11258.36682493142
running_model1 = LinearRegression(n_jobs=4)
running_model1.fit(train_X, train_Y)

val_predictions = running_model1.predict(val_X)
rmse = sqrt(mean_squared_error(val_Y, val_predictions))
nrmse = rmse / (max(val_Y) - min(val_Y))
print(f"NRMSE: {nrmse}\nFirst Five Predictions: {val_predictions[:5]}\n First Five True values: {val_Y.head()}")
