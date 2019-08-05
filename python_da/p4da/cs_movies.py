import pandas as pd
import numpy as np

path = "/Users/dag/github/learning/python_da/p4da/pydata-book-2nd-edition/"

unames = ["user_id", "gender", "age", "occupation", "zip"]
users = pd.read_table(path + "datasets/movielens/users.dat", sep="::",
                      header=None, names=unames)

rnames = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_table(path + "datasets/movielens/ratings.dat", sep="::",
                        header=None, names=rnames)

mnames = ["movie_id", "title", "genres"]
movies = pd.read_table(path + "datasets/movielens/movies.dat", sep="::",
                       header=None, names=mnames)

users[:5]
ratings[:5]
movies[:5]

data = pd.merge(pd.merge(ratings, users, on="user_id"),
                movies, on="movie_id")
data[:5]
data.iloc[0]

# get mean rating by movie and gender
mean_ratings = data.pivot_table("rating",  # What
                                index="title",  # Rows
                                columns="gender",  # Cols
                                aggfunc="mean")  # Content
ratings_by_title = data.groupby("title").size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]  # filter movies
mean_ratings = mean_ratings.loc[active_titles]
mean_ratings
# see top movies for gender F
top_female_ratings = mean_ratings.sort_values(by="F", ascending=False)
top_female_ratings
# find top divisive movies
delta_by_gender = mean_ratings["M"] - mean_ratings["F"]
delta_by_gender.name = "diff"
mean_ratings = pd.concat([mean_ratings, delta_by_gender], axis=1)
top_diff_by_gender = mean_ratings.sort_values(by="diff", ascending=False)
top_diff_by_gender

# overall most divisive movies
group by movie id
method sd() on grouped data
sort by sd, ascending = False
mov_std = data.groupby("title")["rating"].std()
mov_std.name = "rating_std"
mov_std.sort_values(ascending=False)  # too much going on
mov_std.loc[active_titles].sort_values(ascending=False)  # reduce to active titles
