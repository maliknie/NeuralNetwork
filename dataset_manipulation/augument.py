import numpy as np
import pandas as pd
import os


FILEPATH_SRC = "datasets/new_train_set.csv"
FILRPATH_DST = "datasets/new_train_set_augmented.csv"

df = pd.read_csv(FILEPATH_SRC)
target = df["label"]
data = df.drop("label", axis=1)

def brightness(src, dst, factor, d):

    random_d = np.random.randint(0, d)/200

    df = pd.read_csv(src)
    
    target = df["label"]
    data = df.drop("label", axis=1)
    
    for i in range(len(data)):
    
        data.iloc[i] = np.clip(data.iloc[i] * (factor + random_d), 0, 255)  

    augmented_df = pd.concat([target, data], axis=1)

    augmented_df.to_csv(dst, index=False)
    print(f"Augmented dataset saved to {dst}")

brightness(FILEPATH_SRC, FILRPATH_DST, 0.9, 50)
