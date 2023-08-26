import os

import numpy as np
import pandas as pd

path_to_folder = r"local_data\records\descriptive_analyses\target-specific\HTX\tables\original"

##########################

a = pd.read_csv(os.path.join(path_to_folder, "full_dataset", "description_cat_features.csv"))
b = pd.read_csv(os.path.join(path_to_folder, "full_dataset", "description_cont_features.csv"))

a["new"] = a["n"].str.extract(r'^(\d+)')
a["new"] = a["new"] + " (" + a['%'].astype(str) + ")"
print(a)

age_df = b[b["Unnamed: 0"].isin(["AGE"])]
psa_df = b[b["Unnamed: 0"].isin(["PSA"])]

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

ds = pd.DataFrame()

ds["0"] = pd.concat([age_df["0"], psa_df["0"], a["new"]], ignore_index=True)

############################

a = pd.read_csv(os.path.join(path_to_folder, "full_dataset", "target_description_cat_features.csv"))
b = pd.read_csv(os.path.join(path_to_folder, "full_dataset", "target_description_cont_features.csv"))

a["neg"] = a["Negative n/N"].str.extract(r'^(\d+)')
a["neg"] = a["neg"] + " (" + a['Negative %'].astype(str) + ")"

a["pos"] = a["Positive n/N"].str.extract(r'^(\d+)')
a["pos"] = a["pos"] + " (" + a['Positive %'].astype(str) + ")"

age_df = pd.DataFrame(b.iloc[2]).transpose()
psa_df = pd.DataFrame(b.iloc[4]).transpose()

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_neg_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_neg_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

age_df = pd.DataFrame(b.iloc[3]).transpose()
psa_df = pd.DataFrame(b.iloc[5]).transpose()

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_pos_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_pos_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

ds["1"] = pd.concat([age_neg_df["0"], psa_neg_df["0"], a["neg"]], ignore_index=True)
ds["2"] = pd.concat([age_pos_df["0"], psa_pos_df["0"], a["pos"]], ignore_index=True)

p_value_dataset = pd.DataFrame({"p-value": [
    b["p-value"].iloc[2], np.nan, np.nan, b["p-value"].iloc[4], np.nan, np.nan,
    a["p-value"].iloc[0], np.nan, np.nan, a["p-value"].iloc[2], np.nan, np.nan, np.nan,
    a["p-value"].iloc[5], np.nan, np.nan, np.nan, a["p-value"].iloc[8], np.nan, np.nan, np.nan,
]}).round(4)

############################

a = pd.read_csv(os.path.join(path_to_folder, "train", "description_cat_features.csv"))
b = pd.read_csv(os.path.join(path_to_folder, "train", "description_cont_features.csv"))

a["new"] = a["n"].str.extract(r'^(\d+)')
a["new"] = a["new"] + " (" + a['%'].astype(str) + ")"

age_df = b[b["Unnamed: 0"].isin(["AGE"])]
psa_df = b[b["Unnamed: 0"].isin(["PSA"])]

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

ds["4"] = pd.concat([age_df["0"], psa_df["0"], a["new"]], ignore_index=True)

############################

a = pd.read_csv(os.path.join(path_to_folder, "train", "target_description_cat_features.csv"))
b = pd.read_csv(os.path.join(path_to_folder, "train", "target_description_cont_features.csv"))

a["neg"] = a["Negative n/N"].str.extract(r'^(\d+)')
a["neg"] = a["neg"] + " (" + a['Negative %'].astype(str) + ")"

a["pos"] = a["Positive n/N"].str.extract(r'^(\d+)')
a["pos"] = a["pos"] + " (" + a['Positive %'].astype(str) + ")"

age_df = pd.DataFrame(b.iloc[2]).transpose()
psa_df = pd.DataFrame(b.iloc[4]).transpose()

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_neg_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_neg_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

age_df = pd.DataFrame(b.iloc[3]).transpose()
psa_df = pd.DataFrame(b.iloc[5]).transpose()

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_pos_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_pos_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

ds["5"] = pd.concat([age_neg_df["0"], psa_neg_df["0"], a["neg"]], ignore_index=True)
ds["6"] = pd.concat([age_pos_df["0"], psa_pos_df["0"], a["pos"]], ignore_index=True)

p_value_learning_set = pd.DataFrame({"p-value": [
    b["p-value"].iloc[2], np.nan, np.nan, b["p-value"].iloc[4], np.nan, np.nan,
    a["p-value"].iloc[0], np.nan, np.nan, a["p-value"].iloc[2], np.nan, np.nan, np.nan,
    a["p-value"].iloc[5], np.nan, np.nan, np.nan, a["p-value"].iloc[8], np.nan, np.nan, np.nan,
]}).round(4)

############################

a = pd.read_csv(os.path.join(path_to_folder, "test", "description_cat_features.csv"))
b = pd.read_csv(os.path.join(path_to_folder, "test", "description_cont_features.csv"))

a["new"] = a["n"].str.extract(r'^(\d+)')
a["new"] = a["new"] + " (" + a['%'].astype(str) + ")"

age_df = b[b["Unnamed: 0"].isin(["AGE"])]
psa_df = b[b["Unnamed: 0"].isin(["PSA"])]

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

ds["8"] = pd.concat([age_df["0"], psa_df["0"], a["new"]], ignore_index=True)
print(ds)

############################

a = pd.read_csv(os.path.join(path_to_folder, "test", "target_description_cat_features.csv"))
b = pd.read_csv(os.path.join(path_to_folder, "test", "target_description_cont_features.csv"))

a["neg"] = a["Negative n/N"].str.extract(r'^(\d+)')
a["neg"] = a["neg"] + " (" + a['Negative %'].astype(str) + ")"

a["pos"] = a["Positive n/N"].str.extract(r'^(\d+)')
a["pos"] = a["pos"] + " (" + a['Positive %'].astype(str) + ")"

age_df = pd.DataFrame(b.iloc[2]).transpose()
psa_df = pd.DataFrame(b.iloc[4]).transpose()

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_neg_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_neg_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

age_df = pd.DataFrame(b.iloc[3]).transpose()
psa_df = pd.DataFrame(b.iloc[5]).transpose()

age_mean, age_median, age_min, age_max = age_df["mean"].values[0], age_df["50%"].values[0], age_df["min"].values[0], age_df["max"].values[0]
psa_mean, psa_median, psa_min, psa_max = psa_df["mean"].values[0], psa_df["50%"].values[0], psa_df["min"].values[0], psa_df["max"].values[0]

age_pos_df = pd.DataFrame({"0": [f"{age_mean} ({age_median})", f"{age_min} - {age_max}"]})
psa_pos_df = pd.DataFrame({"0": [f"{psa_mean} ({psa_median})", f"{psa_min} - {psa_max}"]})

ds["9"] = pd.concat([age_neg_df["0"], psa_neg_df["0"], a["neg"]], ignore_index=True)
ds["10"] = pd.concat([age_pos_df["0"], psa_pos_df["0"], a["pos"]], ignore_index=True)

p_value_holdout_set = pd.DataFrame({"p-value": [
    b["p-value"].iloc[2], np.nan, np.nan, b["p-value"].iloc[4], np.nan, np.nan,
    a["p-value"].iloc[0], np.nan, np.nan, a["p-value"].iloc[2], np.nan, np.nan, np.nan,
    a["p-value"].iloc[5], np.nan, np.nan, np.nan, a["p-value"].iloc[8], np.nan, np.nan, np.nan,
]}).round(4)

############################

idx_list = [-1, 1, 3, 5, 8, 11]


# Function to insert rows with NaN values at specific indices
def insert_nan_rows(df, idx_list):
    new_rows = pd.DataFrame(np.nan, columns=df.columns, index=idx_list)
    combined_df = pd.concat([df, new_rows]).sort_index().reset_index(drop=True)
    return combined_df.reindex(range(len(combined_df)))


# Call the function to add rows with NaN at specified indices
ds = insert_nan_rows(ds, idx_list)

ds["3"] = p_value_dataset
ds["7"] = p_value_learning_set
ds["11"] = p_value_holdout_set

ds["-1"] = [
    r"Age $[\text{years}]$",
    r"~~~~Mean (Median)",
    r"~~~~Min - Max",
    r"PSA $[\text{ng/ml}]$",
    r"~~~~Mean (Median)",
    r"~~~~Min - Max",
    r"Clinical stage",
    r"~~~~T1-T2",
    r"~~~~T3a",
    r"Global Gleason",
    r"~~~~8",
    r"~~~~9",
    r"~~~~10",
    r"Primary Gleason",
    r"~~~~3",
    r"~~~~4",
    r"~~~~5",
    r"Secondary Gleason",
    r"~~~~3",
    r"~~~~4",
    r"~~~~5",
]
ds = ds[["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]]

print(ds.fillna("").to_latex(index=False, escape=False))
