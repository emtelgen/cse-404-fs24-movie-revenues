import pandas as pd
import matplotlib.pyplot as plt
import joblib  # To save the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from letterbox import Letterboxd
from revenue import RevenueData

def merge_data(letterboxd, revenue_data):
    """
    Merges the Letterboxd and revenue data based on movie names.
    Processes missing data, creates required features, and encodes non-integer variables.
    """
    print('Merging data...')
    #merged_data = letterboxd.merge_data()
    merged_data = pd.read_csv("letterboxdata.csv", nrows=600000)  # Assuming `merge_data` is a method from Letterboxd
    print(merged_data.columns)
    
    # Renaming revenue data for proper merging
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    
    # Merge revenue data with the merged data on 'name'
    merged_data = merged_data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    print('Merged data')

    # Drop rows where revenue is missing
    merged_data = merged_data.dropna(subset=['revenue'])

    # Handling missing values in the 'rating' and 'minute' columns
    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)
    merged_data.to_csv('merged_data.csv', index=False)
    print('Merged data saved to merged_data.csv')
    print(merged_data.dtypes)

    merged_data = merged_data.drop(columns=['name', 'tagline', 'name_crew', 'studio'])

    # Automatically detect categorical columns based on their data type
    categorical_columns = merged_data.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {categorical_columns}")
    merged_data.to_csv('merged_data.csv', index=False)

    # One-Hot Encoding for categorical columns
   # merged_data['cleaned_description'] = merged_data['description'].apply(preprocess_text)
    merged_data = pd.get_dummies(merged_data, columns=['genre', 'language', 'country'], drop_first=True)
    print(merged_data.head)

    # Preprocess the 'description' column and add cleaned description as a new column

    return merged_data

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def train_and_evaluate_model(merged_data):
    """
    Trains a GradientBoostingRegressor model on the merged data.
    Evaluates the model performance and displays the results.
    """
    # Features (X) and target variable (y)
    X = merged_data.drop(columns=['revenue', 'description'])  # Drop unnecessary columns
    y = merged_data['revenue']

    print("Columns: ", X.columns)

    # Convert data to numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': np.arange(50, 200, 10),    # Number of trees in the forest
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],  # Learning rate
        'max_depth': np.arange(3, 15, 1),  # Maximum depth of each tree
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Fraction of samples used for training each tree
        'min_samples_split': np.arange(2, 10, 1),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': np.arange(1, 10, 1)  # Minimum number of samples required at each leaf node
    }

    # Initialize the model
    model = GradientBoostingRegressor(random_state=42)

    # Perform RandomizedSearchCV with 5-fold cross-validation
    randomized_search = RandomizedSearchCV(model, param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1)

    # Fit the RandomizedSearchCV model
    randomized_search.fit(X_train, y_train)

    # Get the best model from the randomized search
    best_model = randomized_search.best_estimator_

    print(f"Best hyperparameters: {randomized_search.best_params_}")

    # Make predictions using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error: {mae}')
    print(f'R^2 Score: {r2}')

    # Plotting predictions vs actual revenue
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    
    # Plotting the ideal 1:1 line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

    plt.title('Model Prediction vs Actual Revenue')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.show()

    # Save the best model from the randomized search
    joblib.dump(best_model, 'gradient_boosting_model_best.pkl')
    print("Best model saved to 'gradient_boosting_model_best.pkl'")

def main():
    """
    Main function to load data, merge it, and train the model.
    """
    base_path = "data"
    
    # Initialize data loaders
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    # Load data
    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    # Merge the data
    merged_data = merge_data(letterboxd, revenue_data)


    # Train and evaluate the model
    train_and_evaluate_model(merged_data)

if __name__ == "__main__":
    main()
