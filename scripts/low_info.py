import os
import pandas as pd

dir = 'download'
less_than_5k = filter(lambda x: os.path.getsize(os.path.join(dir, x)) < 5632,
                      [x for x in os.listdir(dir)])
less = [i[:-4] for i in less_than_5k]
print(len(less))
comparisons = pd.read_csv("../data/safety_votes_overfit.csv")
filtered_final = comparisons[~comparisons["right_id"].isin(less)
                             & ~comparisons["left_id"].isin(less)]
print(len(comparisons), len(filtered_final))
filtered_final.to_csv("../data/safety_votes_filtered.csv", index=False)
