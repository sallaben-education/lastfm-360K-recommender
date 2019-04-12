#!/Users/sallaben/anaconda3/bin/python
print("LOADING LIBRARIES...")
from surprise import KNNBasic
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import Dataset
from surprise import SVD
import pandas as pd

print("LOADING DATASET...")
ratings = pd.read_csv("lastfm-dataset-360K/o0.tsv", sep="\t",
                    names=['user', 'item', 'plays', 'rating'], header=None)

scale = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(ratings[['user', 'item', 'rating']], scale)

print("BUILDING TRAINSET...")
trainset = data.build_full_trainset()
print("FITTING ALGORITHM...")
algo = KNNBasic()
algo.fit(trainset)

# raw user id (as in the ratings file). They are **strings**!
uid = "000063d3fe1cf2ba248b9e3c3f0334845a27a6bf"
iid = "nujabes"  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
print("PREDICTING...")
pred = algo.predict(uid, iid, verbose=True)

print("CROSS VALIDATION...")
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
