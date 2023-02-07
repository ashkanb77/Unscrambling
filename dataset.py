from torch.utils.data import Dataset


class UnscramblingDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __getitem__(self, item):
        row = self.sentences[item]
        return (
           row[0], row[1]
        )

    def __len__(self):
        return len(self.sentences)


