import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(train_informal_path):
    train_df = pd.read_csv(train_informal_path)
    train_df.rename(columns={"formalForm": "FormalForm"}, inplace=True)
    train_df = train_df[['FormalForm']]

    train_df.dropna(inplace=True)

    train_df, val_df = train_test_split(train_df, test_size=0.1)
    # val_df = pd.read_csv(val_informal_path)
    # val_df = val_df[['inFormalForm', 'FormalForm']]
    # val_df.dropna(inplace=True)

    return train_df.values, val_df.values


def collate_fn(data, tokenizer):
    inputs, outputs = zip(*data)
    inputs = list(inputs)
    outputs = list(outputs)

    tokenized = tokenizer(inputs, text_target=outputs, padding=True, return_tensors='pt')
    return tokenized
