import pandas as pd
import matplotlib.pyplot as plt
import joblib  # To save the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import mean_absolute_error, r2_score
from letterbox import Letterboxd
from revenue import RevenueData
from sklearn.tree import DecisionTreeRegressor



def process_data_and_train_model(letterboxd, revenue_data):
    print('Merging data')
    merged_data = letterboxd.merge_data_no_actor() 
    print(merged_data.columns)
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    merged_data = merged_data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    print('Merged data')

    merged_data = merged_data.dropna(subset=['revenue'])

    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)

    X = merged_data.drop(columns=['revenue'])
    y = merged_data['revenue']

    print("Columns: ", X.columns)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a decision tree regressor
    model = DecisionTreeRegressor(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha": [0.0, 0.01, 0.1],
    }

    # Set up the grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",  # Use MSE as the scoring metric
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1  # Use all available cores
    )

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions training data
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print(f'Train Mean Absolute Error: {mae}')
    print(f'Train R^2 Score: {r2}')


    # Predictions real
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Test Mean Absolute Error: {mae}')
    print(f'Test R^2 Score: {r2}')

    plt.figure(figsize=(10, 6))

    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    
    # Plotting the ideal 1:1 line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

    plt.title('Model Prediction vs Actual Revenue')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.show()

    joblib.dump(model, 'decision_tree_model.pkl')
    print("Model saved to 'decision_tree_model.pkl'")

if __name__ == "__main__":
    base_path = "data"
    
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    process_data_and_train_model(letterboxd, revenue_data)
