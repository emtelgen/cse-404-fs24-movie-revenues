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

    def merge_data(self):
        """Merges all the datasets into one."""
        #merged_data = pd.merge(self.movies, self.actors, on='id', how='left')
        
        merged_data = pd.merge(self.movies, self.crew, on='id', how='left', suffixes=('', '_crew'))
        
        # Merge with languages and add suffix '_languages'
        merged_data = pd.merge(merged_data, self.languages, on='id', how='left', suffixes=('', '_languages'))
        
        # Merge with studios and add suffix '_studios'
        merged_data = pd.merge(merged_data, self.studios, on='id', how='left', suffixes=('', '_studios'))
        
        merged_data = pd.merge(merged_data, self.countries, on='id', how='left')
        
        merged_data = pd.merge(merged_data, self.genres, on='id', how='left')
        
        #merged_data = pd.merge(merged_data, self.themes, on='id', how='left')
        
        #merged_data = pd.merge(merged_data, self.releases, on='id', how='left')
        
        return merged_data

    def export_movie_titles(self, filename):
        """Exports movie titles to a CSV file."""
        if self.movies is not None:
            titles_df = self.movies[['name']]  # Extracting the 'name' column
            titles_df.to_csv(filename, index=False)
            print(f"Movie titles exported to {filename}")
        else:
            print("Movies data not loaded.")
