import pandas as pd

class Letterboxd:
    def __init__(self, base_path):
        self.base_path = base_path
        self.movies = None
        self.actors = None
        self.crew = None
        self.languages = None
        self.studios = None
        self.countries = None
        self.genres = None
        self.themes = None
        self.releases = None

    def load_data(self):
        """Loads data from CSV files into pandas DataFrames."""
        self.movies = pd.read_csv(f"{self.base_path}/movies_updated.csv.zip")
        self.actors = pd.read_csv(f"{self.base_path}/actors.csv.zip")
        self.crew = pd.read_csv(f"{self.base_path}/crew.csv.zip")
        self.languages = pd.read_csv(f"{self.base_path}/languages.csv.zip")
        self.studios = pd.read_csv(f"{self.base_path}/studios.csv.zip")
        self.countries = pd.read_csv(f"{self.base_path}/countries.csv.zip")
        self.genres = pd.read_csv(f"{self.base_path}/genres.csv.zip")
        self.themes = pd.read_csv(f"{self.base_path}/themes.csv.zip")
        self.releases = pd.read_csv(f"{self.base_path}/releases.csv.zip")

        print("Letterboxd data loaded successfully.")

    def export_movie_titles(self, filename):
        """Exports movie titles to a CSV file."""
        if self.movies is not None:
            titles_df = self.movies[['name']]  # Extracting the 'name' column
            titles_df.to_csv(filename, index=False)
            print(f"Movie titles exported to {filename}")
        else:
            print("Movies data not loaded.")
