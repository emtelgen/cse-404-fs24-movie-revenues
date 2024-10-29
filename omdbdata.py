import requests
import pandas as pd

class OMDBDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://www.omdbapi.com/"
        self.data = []

    def fetch_movie_data(self, title):
        """Fetch movie data from OMDb API."""
        params = {
            't': title,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data for {title}: {response.status_code}")
            return None

    def collect_data(self, movie_titles):
        """Collect data for a list of movie titles."""
        for title in movie_titles:
            movie_data = self.fetch_movie_data(title)
            if movie_data and 'Response' in movie_data and movie_data['Response'] == 'True':
                self.data.append(movie_data)
            else:
                print(f"Movie not found or error: {title}")

    def export_to_csv(self, filename):
        """Export collected data to a CSV file."""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")

    def load_movie_titles_from_csv(self, filename):
        """Load the first 1000 movie titles from a CSV file."""
        df = pd.read_csv(filename, nrows=1000)  # Load only the first 1000 rows
        return df['name'].tolist()  # Assuming the column containing titles is named 'name'

if __name__ == "__main__":
    # Replace with your own OMDb API key
    API_KEY = "e0c45287"

    # Load movie titles from the CSV file
    collector = OMDBDataCollector(API_KEY)
    movie_titles = collector.load_movie_titles_from_csv("movie_titles.csv")  # Adjust the filename as needed

    # Collect data for the loaded movie titles
    collector.collect_data(movie_titles)
    collector.export_to_csv("omdb_movies_data.csv")
