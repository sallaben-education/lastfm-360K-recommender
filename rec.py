#!/Users/sallaben/anaconda3/bin/python
print("LOADING LIBRARIES...")
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, user_knn as knn
from lenskit import topn
import pandas as pd

print("LOADING DATASET...")
#ratings = pd.read_csv("lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv", sep="\t",
#                      names = ['user', 'artistid', 'artist', 'plays'], header = None)
#profiles = pd.read_csv("lastfm-dataset-360K/usersha1-profile.tsv", sep="\t",
#                      names=['user', 'gender', 'age', 'country', 'signup'], header=None)
ratings = pd.read_csv("lastfm-dataset-360K/obsession3.tsv", sep="\t",
                    names=['user', 'item', 'plays', 'rating'], header=None)
# ratings = ratings.drop(ratings.index[[12520108,14801833]])

#algo_ii = knn.ItemItem(50)
#algo_uu = knn.UserUser(50)
algo_als = als.BiasedMF(50)

# print("SUMMING PLAYS BY USER...")
# plays = ratings.groupby(['user'], as_index=False)[['user', 'plays']].sum()
# plays.reset_index()
# playdict = dict(zip(plays["user"], plays["plays"]))

# print("CALCULATING LEVEL OF OBSESSION...")
# obsession = []
# for index, row in ratings.iterrows():
#     if (index % 100000) == 0:
#         print(index)
#     obsession.append(round(float(row["plays"] / playdict[row["user"]]), 5))

# print("EXPORTING TO CSV...")
# ratings = ratings.assign(o=obsession)
# del ratings["artistid"]
# ratings.to_csv("lastfm-dataset-360K/obsession3.tsv", sep="\t", header=None)

def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 25)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

print("FITTING AND EVALUATING...")
all_recs = []
test_data = []
for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
    test_data.append(test)
    #all_recs.append(eval('UserUser', algo_uu, train, test))
    #all_recs.append(eval('ItemItem', algo_ii, train, test))
    all_recs.append(eval('ALS', algo_als, train, test))
all_recs = pd.concat(all_recs, ignore_index=True)

print(all_recs.head())
