import torch
import torch.utils.data as Data
from collections import Counter
from tqdm import tqdm

class Text_Dataset(Data.Dataset):
    def __init__(self, filename, n_size, vocab_size=1024, encoding="gbk", type="n_gram"):
        self.n_size = n_size
        self.data_process(filename=filename, vocab_size=vocab_size, save=False)

        with open(filename, encoding=encoding) as f:
            # used to construct n-gram
            lines = f.readlines()

        x = []
        y = []
        if type == "n_gram":
            # n-gram
            for line in lines:
                words = line.split()
                if len(words) >= n_size:
                    for i in range(len(words) - n_size + 1):
                        x.append([self.top_words.get(word, 0) for word in words[i:i + n_size - 1]])
                        y.append(self.top_words.get(words[i + n_size - 1], 0))
        elif type == "rnn":
            # for rnn
            for line in lines:
                words = line.split()
                if len(words) >= n_size:
                    for i in range(len(words) - n_size + 1):
                        x.append([self.top_words.get(word, 0) for word in words[i:i + n_size - 1]])
                        y.append([self.top_words.get(word, 0) for word in words[i + 1:i + n_size]])
        else:
            assert False, "Unknown Type"

        self.x_tensor = torch.tensor(x)
        self.y_tensor = torch.tensor(y)

        self.len = len(self.x_tensor)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]

    def data_process(self, filename, vocab_size=1024, save=False):
        # 读取文本文件
        with open(filename, 'r') as file:
            lines = file.readlines()

        with open(filename, 'r') as f:
            # used to select the top common words
            words = f.read().split()

        # 统计单词出现的次数
        word_counts = Counter(words)

        # top words
        self.top_words = {word[0]: idx + 1 for idx, word in enumerate(word_counts.most_common(vocab_size - 1))}

        if save:
            with open('words_count.txt', 'w') as f:
                for word, count in word_counts.most_common():
                    f.write(word + ' ' + str(count) + '\n')


