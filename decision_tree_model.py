import pandas as pd
import matplotlib.pyplot as plt
import joblib  # To save the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import mean_absolute_error, r2_score
from letterbox import Letterboxd
from revenue import RevenueData
from sklearn.tree import DecisionTreeClassifier



def process_data_and_train_model(letterboxd, revenue_data):
    print('Merging data')
    merged_data = letterboxd.merge_data() 
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
    model = DecisionTreeClassifier()
    maes = []
    r2s = []
    splits = [.9, .8, .7, .6, .5, .4, .3, .2]
    for split in (splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)

        #model = DecisionTreeClassifier()

        # Train the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))

    print(f'Mean Absolute Error: {maes[-1]}')
    print(f'R^2 Score: {r2s[-1]}')

    plt.figure(figsize=(10, 6))

    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    
    # Plotting the ideal 1:1 line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

    plt.title('Model Prediction vs Actual Revenue')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.show()
    plt.savefig("decision_tree_predictions")

    # MAE by training size
    plt.figure(figsize=(10, 6))
    plt.scatter(splits, maes, color="blue", alpha=-0.6)
    
    plt.title('MAE over training size')
    plt.xlabel('Split of test data')
    plt.ylabel('MAE')
    plt.show()
    plt.savefig("decision_tree_mae")

    # R2 by training size
    plt.figure(figsize=(10, 6))
    plt.scatter(splits, r2s, color="blue", alpha=-0.6)
    
    plt.title('R2 over training size')
    plt.xlabel('Split of test data')
    plt.ylabel('R2')
    plt.show()
    plt.savefig("decision_tree_r2")

    #joblib.dump(model, 'gradient_boosting_model.pkl')
    #print("Model saved to 'gradient_boosting_model.pkl'")

if __name__ == "__main__":
    base_path = "data"
    
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    process_data_and_train_model(letterboxd, revenue_data)
