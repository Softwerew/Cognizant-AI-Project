#Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest,f_regression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load data
   df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Create target variable and predictor variables
def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y
    
    

# Train algorithm
def train_algorithm_with_cross_validation(
    X: pd.DataFrame = None, 
    y: pd.Series = None
):
    """
    This function takes the predictor and target variables and
    trains a Random Forest Regressor model across K folds. Using
    cross-validation, performance metrics will be output for each
    fold during training.

    :param      X: pd.DataFrame, predictor variables
    :param      y: pd.Series, target variable

    :return
    """
    
    # stardadizing the features using Standard scaler to bring features on a similar scale.             	sc = StandardScaler()
		sc.fit_transform(X)
		
	# splitting the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=2)
        
    #Model Trainining
    # Instantiate algorithm
        model 1= SupportVectorRegressor()
        model 2=RandomForestRegressor()
        model 3=GradientBoostingRegressor()

        
    
        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using R2 Score,Adjusted R2 score and mean absolute error
        R2 score=metrics.r2_score(y_test,y_pred)
        #n = number of rows
        #k = number of features
        Adj_score=1 - (1 - R2 Score)*(n-1) / (n-k-1)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        
        #K-Fold Validation
         X_ml=pd.concat([X_train, X_test])
         y_ml = pd.concat([y_train, y_test])
         all_ml = KFold(n_splits=20, shuffle=True, random_state=2)
         
        #Cross Validation on RandomForest model
        score_all_mae= cross_val_score(rf, X_ml, y_ml, cv=all_ml, scoring='neg_mean_absolute_error')
    # Finish by computing the average MAE across all folds
    print('Mean score over all folds: ', score_all_mae.mean())