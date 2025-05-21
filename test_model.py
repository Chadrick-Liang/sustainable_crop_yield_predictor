from typing import TypeAlias
from typing import Optional, Any    

Number: TypeAlias = int | float

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from IPython.display import display

#Mathematical Functions
def normalize_z(array: np.ndarray, 
                columns_means: Optional[np.ndarray]=None, 
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if columns_means is None:
        columns_means=np.mean(array,axis=0)
    if columns_stds is None:
        columns_stds=np.std(array,axis=0)
        
    out = (array-columns_means)/columns_stds
    return out, columns_means, columns_stds

def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_feature = pd.DataFrame(df[feature_names[:]])
    df_target = pd.DataFrame(df[target_names[:]])
    return df_feature, df_target

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    one_column = np.ones((np_feature.shape[0], 1))
    return np.concatenate((one_column, np_feature), axis=1)

def predict_linreg(array_feature: np.ndarray, beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    array_feature = np.asarray(normalize_z(array_feature, means, stds)[0])
    array_feature = prepare_feature(array_feature)
    return calc_linreg(array_feature, beta)

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.matmul(X, beta)

def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    y_hat = calc_linreg(X, beta)
    #print(f'i am y_hat {y_hat}') y_hat is an array of single value arrays containing one value.
    J = np.sum((y_hat - y)**2) / (2*m)
    #print(np.squeeze(J)) 
    return np.squeeze(J)

def gradient_descent_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray, 
                            alpha: float, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    J_storage = np.zeros((num_iters, 1))
    #print('this is J_storage: ', J_storage)
    m = X.shape[0]
    #print(f'this is m: {m}')
    for i in range(num_iters):
        cost = compute_cost_linreg(X, y, beta)
        if cost == J_storage[-1]:
            print('cost function values stagnated')
            return beta, J_storage
        J_storage[i] = cost
        if i > 0 and abs(cost) > 1000000000:
            print(f"Cost is too large at iteration {i}, stopping early.")
            print(J_storage)
            return beta, J_storage
        y_hat = calc_linreg(X, beta)
        #print(f'this is y_hat: {y_hat}')
        # b <- b - alpha / m . (X^T * (X * b - y))
        beta = beta - (alpha / m) * (np.matmul(X.T, (y_hat - y)))

    return beta, J_storage

def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(random_state)
    
    df_feature_train=df_feature.copy()
    df_target_train=df_target.copy()

    arridx=np.random.choice(df_feature.shape[0],int(test_size*df_feature_train.shape[0]),replace=False)

    df_feature_test=df_feature_train.loc[arridx]
    df_feature_train.drop(arridx,inplace=True)
    
    df_target_test=df_target_train.loc[arridx]
    df_target_train.drop(arridx,inplace=True)
    return df_feature_train, df_feature_test, df_target_train, df_target_test

def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    y_mean = np.mean(y)
    SStot = np.sum((y - y_mean) ** 2)
    SSres = np.sum((y - ypred) ** 2)
    r2 = 1 - (SSres / SStot)
    return r2

def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
    MSE = np.mean((target - pred) ** 2)
    return MSE

def graph_end_relationship(actual, pred):
    pred = np.sort(pred, axis=None)
    df_var = actual.copy()
    df_var.sort_values(ascending=True, inplace=True)
    plt.scatter(df_var, pred)
    plt.plot([min(df_var), max(df_var)], [min(pred), max(pred)], color='black')
    plt.xlabel('Actual feature values')
    plt.ylabel('Predicted feature values')
    plt.show()

def transform_features(df_feature: pd.DataFrame, 
                       colname: str, 
                       colname_transformed: str,
                       power) -> pd.DataFrame:
    #df_feature[colname_transformed] = np.square(df_feature[colname])
    df_feature[colname_transformed] = np.power(df_feature[colname], power)
    return df_feature


#Extracting relevant data from all datasets
yield_df: pd.DataFrame = pd.read_csv("yield.csv")
yield_df = yield_df.iloc[:-3, 2:4]
yield_df.rename(columns={
    'Year': 'Year',
    'Cereals | 00001717 || Yield | 005419 || tonnes per hectare': 'Crop Yield'
}, inplace=True)
#print(yield_df)
#print(max(yield_df['Crop Yield']) - min(yield_df['Crop Yield']))

machine_df: pd.DataFrame = pd.read_csv("machinery-per-agricultural-land.csv")
machine_use_df = machine_df[(machine_df['Entity'] == 'Indonesia') & (machine_df['Year'] <= 2019)]
machine_use_df = machine_use_df.loc[:, ['Year','machinery_per_ag_land']]
machine_use_df.rename(columns={
    'Year': 'Year',
    'machinery_per_ag_land': 'Machinery Use Per Area'
}, inplace=True)
#print(machine_use_df)

co2_df: pd.DataFrame = pd.read_csv("co2.csv")
co2_df = co2_df.iloc[72:131, 2:4]
co2_df.rename(columns={
    'Year': 'Year',
    'Annual CO₂ emissions': 'CO2 Emission'
}, inplace=True)
#print(co2_df)

rain_df: pd.DataFrame = pd.read_csv("rain.csv")
rain_df = rain_df.iloc[:, 2:4]
#print(rain_df)



#Combining all relevant data into one dataframe
combined_df = rain_df.merge(machine_use_df, on='Year').merge(co2_df, on='Year').merge(yield_df, on='Year')
#print(combined_df)



#Splitting columns into feature and target dataframes
df_features, df_target = get_features_targets(combined_df, ["Annual precipitation", "Machinery Use Per Area", "CO2 Emission"], ["Crop Yield"])

#print(df_features[:5])
#print(df_target[:2])

#df_features = transform_features(df_features, "Annual precipitation", "Annual precipitation", 3)
#df_features = transform_features(df_features, "Machinery Use Per Area", "Machinery Use Per Area", 3)
#df_features = transform_features(df_features, "CO2 Emission", "CO2 Emission", 0.5)



#Randomly splitting the targets and features into training dataframes and testing dataframes
df_feature_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state=100, test_size=0.3)
#normalizing the features and extracting stats
array_features_train_z, means, stds = normalize_z(df_feature_train)
print(df_feature_train[:5])
print(f'these are the means {means}')
print(f'these are the stds {stds}')

#converting target into numpy array
X: np.ndarray = prepare_feature(array_features_train_z)
target: np.ndarray = df_target_train.to_numpy()

#print(X[:5])

#Gradient descent to find coefficents beta and lowest cost function
iterations: int = 1500
alpha: float = 0.01
beta: np.ndarray = np.zeros((4,1))
# Call the gradient_descent function
beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)
'''plt.plot(J_storage)
plt.show()'''
#print(beta)




# call the predict() method with established coefficients to generate predicted Y_cap values
pred: np.ndarray = predict_linreg(df_features_test, beta, means, stds)
#plotting predicted Y_cap values against actual Y test values
'''plt.scatter(df_features_test["Annual precipitation"], df_target_test)
plt.scatter(df_features_test["Annual precipitation"], pred)'''
'''plt.scatter(df_features_test["Machinery Use Per Area"], df_target_test)
plt.scatter(df_features_test["Machinery Use Per Area"], pred)'''
'''plt.scatter(df_features_test["CO2 Emission"], df_target_test)
plt.scatter(df_features_test["CO2 Emission"], pred)'''
#plt.show()



#Check accuracy of predicted Y_cap values via r^2, MSE and correlation
target: np.ndarray = df_target_test.to_numpy()
r2: float = r2_score(target, pred)
print(f'this is the r2 score {r2}')

mse: float = mean_squared_error(target, pred)
print(f'this is the mean squared error {mse}')

correlation = np.corrcoef(df_target_test.values.flatten(), pred.flatten())[0, 1]
print(f'Correlation between predicted and actual values: {correlation}')

#graph_end_relationship(df_features_test["Machinery Use Per Area"], pred)
#graph_end_relationship(df_features_test["Fertilizer Use Per Area"], pred)
#graph_end_relationship(df_features_test["Arable Land"], pred)

