import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from letterbox import Letterboxd
from revenue import RevenueData

def process_data_and_train_model(letterboxd, revenue_data):
    print('Merging data')
    merged_data = letterboxd.merge_data()  # Aggregated actor data is already in this DataFrame
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    merged_data = merged_data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    print('Merged data')

    # Drop rows where revenue is missing
    merged_data = merged_data.dropna(subset=['revenue'])

    # Fill missing values in numeric columns
    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)

    # Features (X) and target (y)
    X = merged_data.drop(columns=['revenue'])
    y = merged_data['revenue']

    print("Columns: ", X.columns)

    # Convert all columns to numeric, filling errors with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
    print(f'R^2 Score: {r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    base_path = "data"
    
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    process_data_and_train_model(letterboxd, revenue_data)
