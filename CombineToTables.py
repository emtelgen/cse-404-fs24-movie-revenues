import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from letterbox import Letterboxd
from omdbdata import OMDBDataCollector
from revenue import RevenueData

# Load and initialize the data classes (make sure the paths and API keys are correct)
letterboxd = Letterboxd(base_path='path_to_letterboxd_data')
revenue_data = RevenueData(base_path='path_to_revenue_data')
omdb_data = OMDBDataCollector(api_key="YOUR_API_KEY")

# Load Letterboxd, Revenue, OMDb
letterboxd.load_data()
revenue_data.load_box_office_data()
movie_titles = letterboxd.movies['name'].tolist()[:100]
omdb_data.collect_data(movie_titles)  
omdb_table = pd.DataFrame(omdb_data.data)  

# Letterboxd tables
movie_features_table = letterboxd.movies[['id', 'name', 'director', 'release', 'genres', 'language']]
actors_table = letterboxd.actors[['id', 'actor', 'role']]
crew_table = letterboxd.crew[['id', 'crew_member', 'position']]
genres_table = letterboxd.genres[['id', 'genre']]
print("Letterboxd Movie Features Table:")
print(movie_features_table.head())

# Revenue tables
box_office_table = revenue_data.box_office_data[['Title', 'Worldwide', 'Domestic', 'Foreign']]
brand_table = revenue_data.brand_data[['Brand', 'AverageWorldwide']]
franchise_table = revenue_data.franchise_data[['Franchise', 'TotalWorldwide']]
print("\nRevenue Box Office Table:")
print(box_office_table.head())

# OMDb Data table 
print("\nOMDb Data Table:")
print(omdb_table.head())

# Save each table
movie_features_table.to_csv("movie_features_table.csv", index=False)
actors_table.to_csv("actors_table.csv", index=False)
box_office_table.to_csv("box_office_table.csv", index=False)
omdb_table.to_csv("omdb_table.csv", index=False) 

# Merge movie features with box office data on movie title
movies_with_revenue = pd.merge(movie_features_table, box_office_table, left_on='name', right_on='Title', how='inner')
# Merge with actors
movies_with_actors = pd.merge(movies_with_revenue, actors_table, left_on='id', right_on='movie_id', how='left')

# Final merged table
final_table = movies_with_actors  

# Display merged table
print("\nFinal Merged Table:")
print(final_table.head())
print("\nFinal Merged Table Info:")
print(final_table.info())

