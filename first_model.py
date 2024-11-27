import pandas as pd
import matplotlib.pyplot as plt
import joblib  # To save the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from letterbox import Letterboxd
from revenue import RevenueData
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data for text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocesses the text by lowering case, removing punctuation, stopwords, and lemmatizing.
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Lemmatize the tokens (convert to base form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Return cleaned text
    return " ".join(tokens)

def merge_data(letterboxd, revenue_data):
    """
    Merges the Letterboxd and revenue data based on movie names.
    Processes missing data, creates required features, and encodes non-integer variables.
    """
    print('Merging data...')
    merged_data = letterboxd.merge_data()  # Assuming `merge_data` is a method from Letterboxd
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
    
    print(merged_data.dtypes)

    merged_data = merged_data.drop(columns=['name'])

    # Automatically detect categorical columns based on their data type
    categorical_columns = merged_data.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {categorical_columns}")

    # One-Hot Encoding for categorical columns
    merged_data = pd.get_dummies(merged_data, columns=categorical_columns, drop_first=True)

    # Preprocess the 'description' column and add cleaned description as a new column
    merged_data['cleaned_description'] = merged_data['description'].apply(preprocess_text)

    return merged_data

def extract_text_features(merged_data):
    """
    Extracts TF-IDF features from the 'cleaned_description' column.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)

    # Fit and transform the 'cleaned_description' column
    tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['cleaned_description'])

    # Convert the sparse matrix to a DataFrame for easy inspection
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Merge the TF-IDF features with the original data
    merged_data = pd.concat([merged_data, tfidf_df], axis=1)

    return merged_data

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
from scipy.stats import uniform, randint

def train_and_evaluate_model(merged_data):
    """
    Trains a GradientBoostingRegressor model on the merged data with randomized search for hyperparameter tuning.
    Evaluates the model performance and displays the results.
    """
    # Features (X) and target variable (y)
    X = merged_data.drop(columns=['revenue', 'description', 'name', 'tagline', 'cleaned_description'])  # Drop unnecessary columns
    y = merged_data['revenue']

    print("Columns: ", X.columns)

    # Convert data to numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the GradientBoostingRegressor model
    model = GradientBoostingRegressor(random_state=42)

    # Define the hyperparameter search space
    param_dist = {
        'n_estimators': randint(50, 200),  # Number of boosting stages
        'learning_rate': uniform(0.01, 0.2),  # Learning rate
        'max_depth': randint(3, 10),  # Maximum depth of each tree
        'min_samples_split': randint(2, 20),  # Minimum number of samples required to split a node
        'min_samples_leaf': randint(1, 20),  # Minimum number of samples required at a leaf node
        'subsample': uniform(0.7, 0.3),  # Fraction of samples used for fitting each tree
        'max_features': ['auto', 'sqrt', 'log2', None]  # Maximum number of features to consider
    }

    # Set up the RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist, 
        n_iter=50,  # Number of random combinations to try
        cv=3,  # 3-fold cross-validation
        verbose=2,  # Verbosity level for progress
        random_state=42, 
        n_jobs=-1  # Use all CPU cores available
    )

    # Perform the randomized search
    randomized_search.fit(X_train, y_train)

    # Get the best model from the randomized search
    best_model = randomized_search.best_estimator_

    print(f"Best parameters: {randomized_search.best_params_}")

    # Make predictions with the best model
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

    # Save the best model
    joblib.dump(best_model, 'gradient_boosting_model_tuned.pkl')
    print("Best model saved to 'gradient_boosting_model_tuned.pkl'")



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

    # Extract text features (from 'description' column)
    merged_data = extract_text_features(merged_data)

    # Train and evaluate the model
    train_and_evaluate_model(merged_data)

if __name__ == "__main__":
    main()
