import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(train_informal_path, split=True):
    df = pd.read_csv(train_informal_path)
    df = df.sample(int(len(df) * 0.2))
    df = df[['Shuffled', 'Original']]

    df.dropna(inplace=True)

    if split:
        train_df, val_df = train_test_split(df.values, test_size=0.1, random_state=7)
        return train_df, val_df

    return df.values


def collate_fn(data, tokenizer):
    inputs, outputs = zip(*data)
    inputs = list(inputs)
    outputs = list(outputs)

    tokenized = tokenizer(inputs, text_target=outputs, padding=True, return_tensors='pt')
    return tokenized
