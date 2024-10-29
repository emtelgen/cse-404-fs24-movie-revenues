import pandas as pd
class RevenueData:
    def __init__(self, base_path):
        self.base_path = base_path
        self.box_office_data = None
        self.brand_data = None
        self.franchise_data = None
        self.genre_data = None
        self.top_movies_data = None

    def load_box_office_data(self):
        """Loads box office data from the TSV file."""
        box_office_file = f"{self.base_path}/boxofficemojointernationaltop1000.tsv"
        self.box_office_data = pd.read_csv(box_office_file, sep='\t')
        print("Box Office data loaded successfully.")

    def load_brand_data(self):
        """Loads brand data from the CSV file."""
        brand_file = f"{self.base_path}/bomojobrandindices.csv"
        self.brand_data = pd.read_csv(brand_file)
        print("Brand data loaded successfully.")

    def load_franchise_data(self):
        """Loads franchise data from the TSV file."""
        franchise_file = f"{self.base_path}/boxofficemojotopfranchises.tsv"
        self.franchise_data = pd.read_csv(franchise_file, sep='\t')
        print("Franchise data loaded successfully.")

    def load_genre_data(self):
        """Loads genre data from the TSV file."""
        genre_file = f"{self.base_path}/boxofficemojotopgenres.tsv"
        self.genre_data = pd.read_csv(genre_file, sep='\t')
        print("Genre data loaded successfully.")

    def load_top_movies_data(self):
        """Loads top movies data from the TSV file."""
        top_movies_file = f"{self.base_path}/boxofficemojoustop1000.tsv"
        self.top_movies_data = pd.read_csv(top_movies_file, sep='\t')
        print("Top Movies data loaded successfully.")