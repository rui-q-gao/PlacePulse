"""
A script to do an exploratory data analysis, investigating the distributions,
the occurrences, and other general details.

Author: Rui Gao
Date Modified: Jan 14, 2020
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/safety_votes.csv")
# data_total = pd.read_csv("../data/votes.csv")
print("Winner distribution: \n" + str(data["winner"].value_counts()) + "\n")

# We can see that the dataset is pretty relatively balanced in terms of left
# and right wins, we may want to delete the draws

freq_l = data["left_id"].value_counts()
freq_r = data["right_id"].value_counts()
print("Left frequency statistics: \n"+str(freq_l.describe()) + "\n")
print("Right frequency statistics: \n"+str(freq_r.describe()) + "\n")

freq_l.sort_values()

print(freq_l[:4109].sum())
cum_freq = freq_l.cumsum()
scale = list(range(len(cum_freq)))
plt.plot(scale, cum_freq)
plt.title("Cumulative Frequency of Comparisons for Most Common Images")
plt.xlabel("Number of Images (Sorted)")
plt.ylabel("Cumulative Frequency of Comparisons")
plt.show()

freq_l.plot.hist(bins=34)
plt.title("Distribution of Image Repetition on Left Side")
plt.xlabel("Frequency of Image for Left")
plt.show()

freq_r.plot.hist(bins=34)
plt.title("Distribution of Image Repetition on Right Side")
plt.xlabel("Frequency of Image for Right")
plt.show()

# We can see that some pictures show up an inordinate amount of times, with
# most pictures showing up from 1-5 times. wtf

print("\nNumber of common pictures:",
      len(set(data["left_id"]) & set(data["right_id"])))

# Interestingly, some pictures that were used for the left image were not
# used for the right image. Roughly 95% of the images were common to both.


