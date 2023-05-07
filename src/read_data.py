import pandas as pd

dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)
print(dataset.shape)
print(dataset.head())
