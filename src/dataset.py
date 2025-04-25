from torch.utils.data import Dataset


class TweetDataset(Dataset):
    """
    Dataset for tweets
    """

    def __init__(self, tokenizer, tweets, labels):
        self.tokenizer = tokenizer
        self.tweets = tweets
        self.labels = labels

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets.iloc[idx]
        label = self.labels.iloc[idx]
        return tweet, int(label)
