"""
A script to collect images from the google_streetview api. First the most
common images to appear in the comparisons is found through sorting twice, then
passed to a function to collect images. The list of missing images is used to
filter out the comparisons.

Author: Rui Gao
Date: Jan 19, 2020
"""
import google_streetview.api
import google_streetview.helpers
import pandas as pd
from time import time
import os
from PIL import Image
import datetime


def download_images(locations, coord, LIMIT):
    # Implementation of google_streetview api

    api_args = {
        'size': '224x274',  # max 640x640 pixels
        'location': locations,
        'pitch': '0',
        'key': 'AIzaSyCdy4d2RDe6DPvr2qqh4PWVE6Xz230uDsE'
    }

    api_list = google_streetview.helpers.api_list(api_args)
    t = time()
    results = google_streetview.api.results(api_list)
    elapsed = time() - t
    print(elapsed)
    results.download_links("download")

    # Renaming files, will be renamed to the image ID
    corrupt = []
    for i in range(LIMIT):
        filename = "download/gsv_" + str(i) + ".jpg"
        try:
            im = Image.open(filename)
        except:
            corrupt.append(i)
            continue
        cropped = im.crop((0, 25, 224, 249))
        cropped.save("download/" + coord['id'].iloc[i] + ".jpg")
        os.remove(filename)
    print(corrupt)
    return corrupt


def main():
    # Arbitrary constant LIMIT, solely used for testing
    LIMIT = 1500
    UNIT = 40
    DOWNLOAD = False
    images = pd.read_csv("../data/safety_votes_images.csv")
    comparisons = pd.read_csv("../data/safety_votes_index.csv")
    # Combining both left and right to find the most common photos
    left_id = comparisons.iloc[:, 0:1]
    left_id.columns = ["id"]
    right_id = comparisons.iloc[:, 1:2]
    right_id.columns = ["id"]
    left_right = pd.concat([left_id, right_id], axis=0)

    # Turning the value_counts() series into a dataframe, to be able to sort
    # twice. Sorting twice ensures that the order is consistent, by frequency
    # and then ID.
    most_freq = left_right["id"].value_counts()
    most_freq = most_freq.to_frame()
    most_freq.columns = ['freq']

    # Reindexing to move original index of picture ID into a separate column
    most_freq['id'] = most_freq.index
    most_freq = most_freq.reset_index(drop=True)
    most_freq = most_freq.sort_values(['freq', 'id'], ascending=[False, True])

    merged_most_freq = pd.merge(left=most_freq, right=images, on='id')

    # Taking most common photos, up to LIMIT
    identity_thus_far = merged_most_freq['id'][:LIMIT * UNIT]

    # Taking a sample of the image coordinates to concatenate into one string
    coord = merged_most_freq.iloc[LIMIT * (UNIT - 1):LIMIT * UNIT]
    coord_thus_far = merged_most_freq.iloc[:LIMIT * UNIT]

    locations = coord["coordinates"].str.cat(sep=';')
    corrupted = []

    if DOWNLOAD:
        corrupted = download_images(locations, coord, LIMIT)

    with open('corrupted.txt', 'r') as corrupted_file:
        last_corrupt = corrupted_file.readlines()[-1]

    if int(last_corrupt) < LIMIT * (UNIT - 1):
        with open('corrupted.txt', 'a') as corrupted_file:
            for i in corrupted:
                corrupted_file.write('%s\n' % str(i + LIMIT * (UNIT - 1)))
    with open('corrupted.txt', 'r') as corrupted_file:
        corrupted_accum = [int(i[:-1]) for i in corrupted_file.readlines()[1:]]

    new_coord = coord_thus_far.reset_index(drop=True)
    corrupted_ids = new_coord[new_coord.index.isin(corrupted_accum)]['id']
    print(corrupted_accum)
    print(corrupted_ids)

    match_final = comparisons[comparisons["right_id"].isin(identity_thus_far)
                              & comparisons["left_id"].isin(identity_thus_far)
                              & ~comparisons["right_id"].isin(corrupted_ids)
                              & ~comparisons["left_id"].isin(corrupted_ids)]

    print(match_final["winner"].value_counts())
    print("\n", datetime.datetime.now(), UNIT)

    error = False
    for i in range(len(match_final)):
        left = "download/" + match_final['left_id'].iloc[i] + ".jpg"
        right = "download/" + match_final['right_id'].iloc[i] + ".jpg"
        if not os.path.exists(left):
            print("O dang entry number {0} left doesn't exist - {1}".format(i, left))
            error = True
        if not os.path.exists(right):
            print("O dang entry number {0} right doesn't exist - {1}".format(i, right))
            error = True

    if not error:
        print("Ay everything aight")


    match_final.to_csv("../data/safety_votes_overfit.csv", index=False)
    return 0


if __name__ == '__main__':
    main()
