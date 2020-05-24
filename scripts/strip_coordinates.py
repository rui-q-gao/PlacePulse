"""
A script to take the original voting database, and splitting into two datasets.
One dataset compiles a dataset of all images by the image ID, and their
corresponding coordinates. The second dataset compiles all the votes, with only
the left image ID, the right image ID, and the winner. The first one is for
collecting data, and the second is for constructing the training data for the
neural network

Author: Rui Gao
Date Modified: Jan 14, 2020
"""
import pandas as pd
import numpy as np

# Combine the latitude and longitude into one coordinate separated by ','
data = pd.read_csv("../data/safety_votes.csv")
data = data[data['winner'] != "equal"]
data = data.drop_duplicates()

# Combine latitude and longitude into one pair of coordinates
left_lat = data["left_lat"].round(6).apply(str)
left = left_lat.str.cat(data["left_long"].round(6).apply(str), sep=",")
right_lat = data["right_lat"].round(6).apply(str)
right = right_lat.str.cat(data["right_long"].round(6).apply(str), sep=",")


data = data.drop(columns=data.columns[3:])

# Balancing the left and right decisions
np.random.seed(2)

data_left = data[data['winner'] == "left"].copy()
data_right = data[data['winner'] == "right"].copy()

data_left = data_left.reset_index(drop=True)
data_right = data_right.reset_index(drop=True)
indices = list(range(len(data_right)))
mid = int(len(data)/2)

np.random.shuffle(indices)
data_right_final = data_right[data_right.index.isin(indices[:mid])].copy()

data_right_to_left = data_right[data_right.index.isin(indices[mid:])].copy()
data_right_to_left.loc[:, "winner"] = "left"
data_right_to_left_final = data_right_to_left[['right_id', 'left_id', 'winner']]
data_right_to_left_final.columns = ['left_id', 'right_id', 'winner']

# Combine adjusted dataframes
data_final = pd.concat([data_left, data_right_to_left_final, data_right_final], axis=0)
data_final = data_final.sample(frac=1, random_state=2).reset_index(drop=True)


# Combine the image ID and coordinates of the left and right images
print(data.shape, left.shape)
dataleft = pd.concat([data.iloc[:, 0:1], left.to_frame()], axis=1)
dataleft.columns = ["id", "coordinates"]
dataright = pd.concat([data.iloc[:, 1:2], right.to_frame()], axis=1)
dataright.columns = ["id", "coordinates"]

# Find unique images and coordinates by dropping duplicates
data_comb = pd.concat([dataleft, dataright], axis=0)
data_comb = data_comb.drop_duplicates()
data_final.loc[:, "winner"] = (data_final["winner"] == "right").astype(int)
data_final.to_csv("../data/safety_votes_index2.csv", index=False)
data_comb.to_csv("../data/safety_votes_images2.csv", index=False)
