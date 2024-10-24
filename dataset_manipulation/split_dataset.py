import pandas as pd

FILEPATH = "Data.csv"

def split_dataset(dataset, train_size=0.8):
    train_data = dataset.sample(frac=train_size, random_state=200)
    test_data = dataset.drop(train_data.index)
    return train_data, test_data

train, test = split_dataset(pd.read_csv(FILEPATH))

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)