"""
A script to examine possible disagreement in votes between the same two images

Author: Rui Gao
Date Modified: Jan 14, 2020
"""

import pandas as pd

comparisons = pd.read_csv("../data/safety_votes_index.csv")

comparisons = comparisons.drop(columns=["winner"])
print(comparisons.shape)

comparisons = comparisons.drop_duplicates()
print(comparisons.shape)

comparisons_switch = comparisons[['right_id', 'left_id']]
comparisons_switch.columns = ['left_id', 'right_id']
c = pd.concat([comparisons, comparisons_switch], axis=0)
print(c.shape)

c = c.drop_duplicates()
print(c.shape)
