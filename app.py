#!/usr/local/bin/python3.7
from flask import Flask
from flask import render_template
from collections import defaultdict
from collections import Counter
import pandas as pd
import pickle
from statistics import median

app = Flask(__name__)

print("LOADING DATASET...")
ratings = pd.read_csv("data/p0x3.tsv", sep="\t",
                      names=['user', 'item', 'plays', 'obsession', 'rating'], header=None)

print("SUMMING USERS BY PLAYS...")
plays = ratings.groupby(['user'], as_index=False)[['user', 'plays']].sum().reset_index()
userplays = dict(zip(plays["user"], plays["plays"]))
      
print("LOADING MODEL...")
algo = pickle.load(open('SVDp0x3.sav', 'rb'))

# users =         ["02eeccf9502709d9d0d34ac99ded328d8201be18",
                #  "0197a658d9ea8811877e518dfe399c52dd8a84d4",
                #  "0301157d84b0ccc9a436479d11676eba0882010b",
                #  "025e476b4e8f4aec3bf263c737b65fd856acd0c2",
                #  "006261139d787c1e43b4c69d304f2772367c1005",
                #  "00b18113f5b06f36dfb587f86b8ef4141dff3118",
                #  "0005a18022e9a3df17694fe19f5b90edadda7953",
                #  "00f57a7fe44eb4d0851d62c5f0ddd003ea43c7ae",
                #  "000429493d9716b66b02180d208d09b5b89fbe64",
                #  "03274dae59b1b5bc750a0738f8733b44972ce3c5",
                #  "01760a1afde70737fa4dd70394e23690b3238768",
                #  "0242309977d951b93ba29da5d0bd780bd237d086",
                #  "00554b78cbcf17e234ce8ef6abcc364ed2a2c4f4",
                #  "0056ccd136b6ad6fcf090e2f51afd5cca888e56f"]
users = ["02eeccf9502709d9d0d34ac99ded328d8201be18",
            "0d1f9e9b5c576082e335a2c197a512432f7e896d",
            "000429493d9716b66b02180d208d09b5b89fbe64",
            "01760a1afde70737fa4dd70394e23690b3238768",
            "0420c51a72f0898c3f064121c0636f1ba73eee3d",
            "0c224cd9379518457cecbcdcdc2d10e7ebcad2d3",
            "01ae97e2cc42b5479e0ac3c2a424325c32b2fff7",
            "0de2759e5b8d1abb6fb7604230c07970da79ee8e",
            "0d53c8106fa54369ad725f6d150f09c799fa7e85"]

# topusers = dict(Counter(userplays).most_common(100))
# users = set(set(topusers.keys()).union(set(users)))

rap =           ["eminem",
                 "ludacris",
                 "kanye west",
                 "lil wayne",
                 "jay-z",
                 "gang starr",
                 "nas",
                 "usher",
                 "50 cent",
                 "common",
                 "2pac",
                 "atmosphere",
                 "outkast",
                 "a tribe called quest",
                 "snoop dogg"]
rock =          ["the beatles",
                 "the who",
                 "nirvana",
                 "pearl jam",
                 "queen",
                 "the rolling stones",
                 "jimi hendrix",
                 "aerosmith",
                 "led zeppelin",
                 "pink floyd",
                 "queens of the stone age",
                 "eagles",
                 "lynyrd skynyrd"]
country =       ["taylor swift",
                 "tim mcgraw",
                 "zac brown band",
                 "carrie underwood",
                 "rascal flatts",
                 "kenny chesney",
                 "eric clapton",
                 "garth brooks"]
alt =           ["radiohead",
                 "green day",
                 "coldplay",
                 "muse",
                 "linkin park",
                 "red hot chili peppers",
                 "modest mouse",
                 "the national",
                 "vampire weekend",
                 "mgmt",
                 "sigur rós",
                 "neutral milk hotel",
                 "dido",
                 "blur"]
electronic =    ["boards of canada",
                 "aphex twin",
                 "röyksopp",
                 "depeche mode",
                 "the chemical brothers",
                 "daft punk",
                 "owl city",
                 "deadmau5",
                 "four tet"]
metal =         ["black sabbath",
                 "metallica",
                 "iron maiden",
                 "megadeth",
                 "judas priest",
                 "slayer",
                 "pantera",
                 "avenged sevenfold",
                 "dream theater",
                 "motörhead",
                 "alice in chains",
                 "incubus",
                 "dragonforce"]


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
    Citation:
        https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, int((est - 0.71549) * 100)))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def mean(nums):
    '''
    Citation:
        https://stackoverflow.com/a/43678550
    '''
    return sum(nums, 0.0) / len(nums)
    
def removelist(lis, remove):
    for rem in remove:
        for el in lis:
            if rem[1][0] == el[0]:
                lis.remove(el)
    return lis

def topn(users, n):
    artists = dict()
    for user in users:
        r = ratings[ratings["user"] == user].nlargest(n, 'plays')
        artists[user] = enumerate(zip(r['item'], r['plays']))
    return artists

def solotopn(user, n):
    r = ratings[ratings["user"] == user].nlargest(n, 'plays')
    return enumerate(zip(r['item'], r['plays']))

plays = ratings.groupby(['item'], as_index=False)[['item', 'plays']].sum().reset_index()
playdict = dict(zip(plays["item"], plays["plays"]))

counts = ratings.item.value_counts()
avg = mean(playdict.values())
playdict = dict((k, v) for k, v in playdict.items() if (v > avg) & (counts[k] > 5))

top150 = list(dict(Counter(playdict).most_common(150)).keys())

all_artists = list(playdict.keys())
all_artists.remove("[unknown]")
all_artists.remove("original broadway cast")
all_artists.remove("böhse onkelz")
all_artists.remove("Бумбокс")
all_artists.remove("Агата Кристи")

def predi(user, items):
    return algo.test(zip([user]*len(items), items, [1]*len(items)), verbose=False)

@app.route("/")
def home():
    return render_template('index.html', data=enumerate(users), userdata=topn(users, 8))

@app.route('/user/<string:username>/')
def show_user(username):
    return render_template('user.html', user=username, userdata=solotopn(username, 25))

@app.route('/user/<string:username>/recommend/<string:genre>/<int:n>/')
def show_recommendations_by_genre(username, genre, n):
    l = []
    if genre == 'rap':
        l = rap
    elif genre == 'rock':
        l = rock
    elif genre == 'country':
        l = country
    elif genre == 'alternative':
        l = alt
    elif genre == 'electronic':
        l = electronic
    elif genre == 'top150':
        l = top150
    elif genre == 'metal':
        l = metal
    else:
        l = all_artists
    predictions = predi(username, l)
    top_n = get_top_n(predictions, n)
    novel = removelist(get_top_n(predictions, n)[username], solotopn(username, 25))
    return render_template('recommend.html', user=username, recs=enumerate(top_n[username]), novelrecs=enumerate(novel))
