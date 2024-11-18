import pandas as pd
from sklearn.linear_model import Ridge
from reservoirpy.nodes import Reservoir

# Load the separated input data from CSV files
input_genres_tags_data = pd.read_csv('../data/train/input_genres_tags_data.csv')
input_instruments_tags_data = pd.read_csv('../data/train/input_instruments_tags_data.csv')
input_moods_tags_data = pd.read_csv('../data/train/input_moods_tags_data.csv')

genres_categories_data = pd.read_csv('../data/train/genres_categories_data.csv')
instruments_categories_data = pd.read_csv('../data/train/instruments_categories_data.csv')
moods_categories_data = pd.read_csv('../data/train/moods_categories_data.csv')

# Load the separated output data from CSV files
output_genres_tags_data = pd.read_csv('../data/train/output_genres_tags_data.csv')
output_instruments_tags_data = pd.read_csv('../data/train/output_instruments_tags_data.csv')
output_moods_tags_data = pd.read_csv('../data/train/output_moods_tags_data.csv')

# Split the data into training and testing sets


