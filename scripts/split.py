"""
A script to split the original data by the category. The safety one is used

Author: Rui Gao
Date Modified: Jan 14, 2020
"""
import pandas as pd

data = pd.read_csv("../data/votes.csv")
data_safe = data[data["category"] == "safety"]
data_lively = data[data["category"] == "lively"]
data_beautiful = data[data["category"] == "beautiful"]
data_wealthy = data[data["category"] == "wealthy"]
data_depress = data[data["category"] == "depressing"]
data_boring = data[data["category"] == "boring"]

data_safe.to_csv("../data/safety_votes.csv", index=False)
data_lively.to_csv("../data/lively_votes.csv", index=False)
data_beautiful.to_csv("../data/beautiful_votes.csv", index=False)
data_wealthy.to_csv("../data/wealthy_votes.csv", index=False)
data_depress.to_csv("../data/depress_votes.csv", index=False)
data_boring.to_csv("../data/boring_votes.csv", index=False)
