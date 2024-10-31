import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from letterbox import Letterboxd 
from revenue import RevenueData

def process_data_and_train_model(letterboxd, revenue_data):
    data = letterboxd.movies    
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    merged_data = data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    actors_df = pd.read_csv(f"{letterboxd.base_path}/actors.csv.zip")
    merged_data = merged_data.merge(actors_df[['id', 'name', 'role']], on='id', how='left')
    merged_data = merged_data.dropna(subset=['revenue'])
    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)

    X = merged_data.drop(columns=['revenue'])
    y = merged_data['revenue']

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
    print(f'R^2 Score: {r2_score(y_test, y_pred)}')


if __name__ == "__main__":
    base_path = "data"
    
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    process_data_and_train_model(letterboxd, revenue_data)
