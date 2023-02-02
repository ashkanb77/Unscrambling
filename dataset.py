from torch.utils.data import Dataset
from random import shuffle
from hazm import WordTokenizer


class UnscramblingDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.tokenizer = WordTokenizer()

    def __getitem__(self, item):
        sentence = self.sentences[item][0]
        words = self.tokenizer.tokenize(sentence)
        shuffle(words)
        return (
           ' '.join(words), sentence
        )

    def __len__(self):
        return len(self.df)


