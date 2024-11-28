import pandas as pd
import matplotlib.pyplot as plt
import joblib  
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import numpy as np

def tune_lightgbm_hyperparameters(X_train, y_train, X_val, y_val):
    param_grid = {
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

    model = lgb.LGBMRegressor()
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, 
                                       n_iter=10, scoring='neg_mean_absolute_error', 
                                       cv=3, random_state=42, verbose=1, n_jobs=-1)
    random_search.fit(X_train, y_train)
    print("Best parameters found based on training data: ", random_search.best_params_)

    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f'Validation Set Mean Absolute Error: {val_mae}')
    print(f'Validation Set R^2 Score: {val_r2}')

    return best_model

def process_data_and_train_model_lightgbm(letterboxd, revenue_data):
    print('Merging data')
    merged_data = letterboxd.merge_data() 
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    merged_data = merged_data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    print('Merged data')

    merged_data = merged_data.dropna(subset=['revenue'])
    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)

    X = merged_data.drop(columns=['revenue'])
    y = merged_data['revenue']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = X[['top_actors_count'] + [col for col in X.columns if col != 'top_actors_count']]  
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = tune_lightgbm_hyperparameters(X_train, y_train, X_val, y_val)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Test Set Mean Absolute Error: {mae}')
    print(f'Test Set R^2 Score: {r2}')

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Model Prediction vs Actual Revenue')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.show()

    joblib.dump(model, 'lightgbm_model.pkl')
    print("Model saved to lightgbm_model.pkl")

def plot_feature_importance(model, X):
    feature_importance = model.feature_importance()
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.title("Feature Importance in LightGBM Model")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show()

def plot_learning_curve(model, X_train, y_train, X_test, y_test, train_sizes):
    train_errors = []
    test_errors = []

    for train_size in train_sizes:
        X_train_sub = X_train[:train_size]
        y_train_sub = y_train[:train_size]

        model.fit(X_train_sub, y_train_sub)

        y_train_pred = model.predict(X_train_sub)
        y_test_pred = model.predict(X_test)

        train_error = mean_absolute_error(y_train_sub, y_train_pred)
        test_error = mean_absolute_error(y_test, y_test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors, label="Training error", color='blue')
    plt.plot(train_sizes, test_errors, label="Testing error", color='red')
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Mean Absolute Error")
    plt.title("Learning Curve for LightGBM Model")
    plt.legend()
    plt.show()

def process_data_and_train_model_lightgbm_with_learning_curve(letterboxd, revenue_data):
    print('Merging data')
    merged_data = letterboxd.merge_data() 
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    merged_data = merged_data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    print('Merged data')

    merged_data = merged_data.dropna(subset=['revenue'])
    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)

    X = merged_data.drop(columns=['revenue'])
    y = merged_data['revenue']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = tune_lightgbm_hyperparameters(X_train, y_train, X_val, y_val)

    train_sizes = np.linspace(0.1, 1.0, 10, dtype=int) * len(X_train)
    plot_learning_curve(model, X_train, y_train, X_test, y_test, train_sizes)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Test Set Mean Absolute Error: {mae}')
    print(f'Test Set R^2 Score: {r2}')

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Model Prediction vs Actual Revenue')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.show()

    joblib.dump(model, 'lightgbm_model.pkl')
    print("Model saved to lightgbm_model.pkl")

if __name__ == "__main__":
    base_path = "data"
    
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    process_data_and_train_model_lightgbm_with_learning_curve(letterboxd, revenue_data)
