import pandas as pd
from letterbox import Letterboxd
from revenue import RevenueData

class DataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.letterboxd = Letterboxd(base_path)
        self.revenue_data = RevenueData(base_path)

    def load_all_data(self):
        """Loads all data from Letterboxd and RevenueData."""
        self.letterboxd.load_data()
        self.revenue_data.load_box_office_data()
        self.revenue_data.load_brand_data()
        self.revenue_data.load_franchise_data()
        self.revenue_data.load_genre_data()
        self.revenue_data.load_top_movies_data()
        
    def export_movie_titles(self):
        """Exports movie titles from Letterboxd."""
        self.letterboxd.export_movie_titles("movie_titles.csv")
    
    def preview_letterboxd_data(self):
        """Displays the first few rows of Letterboxd data only."""
        print("Movies Data:")
        print(self.letterboxd.movies.head(), "\n")
        print(f"Total number of movies: {len(self.letterboxd.movies)}\n") 

        print("Actors Data:")
        print(self.letterboxd.actors.head(), "\n")

        print("Crew Data:")
        print(self.letterboxd.crew.head(), "\n")

        print("Languages Data:")
        print(self.letterboxd.languages.head(), "\n")

        print("Studios Data:")
        print(self.letterboxd.studios.head(), "\n")

        print("Countries Data:")
        print(self.letterboxd.countries.head(), "\n")

        print("Genres Data:")
        print(self.letterboxd.genres.head(), "\n")

        print("Themes Data:")
        print(self.letterboxd.themes.head(), "\n")

        print("Releases Data:")
        print(self.letterboxd.releases.head(), "\n")

if __name__ == "__main__":
    base_path = "data"  
    data_processor = DataProcessor(base_path)
    
    # Load all data
    data_processor.load_all_data()
    
    # Export movie titles
    data_processor.export_movie_titles()

    # Preview only the Letterboxd data
    data_processor.preview_letterboxd_data()
