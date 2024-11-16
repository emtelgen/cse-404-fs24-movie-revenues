import pandas as pd
import matplotlib.pyplot as plt
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def process_data_and_train_model(letterboxd, revenue_data):
    print('Merging data')
    merged_data = letterboxd.merge_data() 
    revenue_data_df = revenue_data.top_movies_data.rename(columns={'Movie': 'name', 'Lifetime Gross': 'revenue'})
    merged_data = merged_data.merge(revenue_data_df[['name', 'revenue']], on='name', how='left')
    print('Merged data')

    merged_data = merged_data.dropna(subset=['revenue'])

    merged_data['average_rating'] = merged_data['rating'].fillna(0)
    merged_data['duration'] = merged_data['minute'].fillna(0)

    revenue_median = merged_data['revenue'].median()
    merged_data['revenue_category'] = merged_data['revenue'].apply(lambda x: 1 if x > revenue_median else 0)

    X = merged_data.drop(columns=['revenue', 'revenue_category'])
    y = merged_data['revenue_category']

    print("Columns: ", X.columns)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:\n', report)
    print('Confusion Matrix:\n', confusion)

    joblib.dump(model, 'logistic_regression_model.pkl')
    print("Model saved to logistic_regression_model.pkl")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.title('Predicted vs Actual Revenue Category')
    plt.xlabel('Actual Category')
    plt.ylabel('Predicted Category')
    plt.show()

if __name__ == "__main__":
    base_path = "data"
    
    letterboxd = Letterboxd(base_path)
    revenue_data = RevenueData(base_path)

    letterboxd.load_data()
    revenue_data.load_top_movies_data()

    process_data_and_train_model(letterboxd, revenue_data)
