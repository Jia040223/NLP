import torch
import torch.utils.data as Data
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import Text_Dataset
import os

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_loaders(batch_size=512, valid=0.0, train=0.8, filename="ChineseCorpus199801.txt", n_size=9, vocab_size=1024,
                type="n_gram", encoding="gbk", num_workers=0, pin_memory=False):
    file_path = "./datasets/save_dataset_{0}_{1}_{2}.pth".format(n_size, vocab_size, type)
    if os.path.exists(file_path):
        dataset = torch.load(file_path)
    else:
        dataset = Text_Dataset(filename=filename, n_size=n_size, vocab_size=vocab_size, encoding=encoding, type=type)
        torch.save(dataset, file_path)
    
    train_size = int(train * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = Data.random_split(dataset, [train_size, test_size])

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = Data.DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    
    valid_loader = Data.DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    
    test_loader = Data.DataLoader(testset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader
