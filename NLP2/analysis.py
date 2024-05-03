import torch
import os
import numpy as np
import random
import argparse


def load_lookup_table(model_name):
    if not os.path.exists("checkpoint"):
        assert Exception("the model {0} is not exist".format(model_name))

    pth_list = os.listdir("checkpoint")
    latest_pth = None
    for pth in pth_list:
        if pth.endswith(".pth") and pth.startswith(model_name):
            if latest_pth is None:
                latest_pth = pth
            else:
                id = int(pth.split("-")[-1].split(".")[0])
                latest_id = int(latest_pth.split("-")[-1].split(".")[0])
                if id > latest_id:
                    latest_pth = pth

    if latest_pth is not None:
        print("load model from checkpoint/" + latest_pth)
        lookup_table = torch.load("checkpoint/" + latest_pth)['embedding.weight'].cpu().numpy()
    else:
        assert Exception("the model {0} is not exist".format(model_name))

    return lookup_table


def get_word(top_words, word_idx):
    name_list = [key for key, value in top_words.items() if value == word_idx]
    if len(name_list) > 0:
        return name_list[0]
    else:
        return "<UNK>"


def get_similar_wordidx(lookup_table, word_idx):
    word_vec = lookup_table[word_idx]

    normed_lookup_table = lookup_table / np.linalg.norm(lookup_table, axis=1).reshape((-1, 1))
    normed_word_vec = word_vec / np.linalg.norm(word_vec)
    similarity = np.dot(normed_lookup_table, normed_word_vec)

    order = np.argsort(-similarity)

    return order, similarity


def top_10_similar(top_words, lookup_table, word_idx):
    orders, similarity = get_similar_wordidx(lookup_table, word_idx)
    for idx, order in enumerate(orders[:10]):
        print("{0} :".format(idx), get_word(top_words, order), similarity[order])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The setting of word embedding analysis')
    parser.add_argument('--random', type=bool, default=True, help='random word or not')
    parser.add_argument('--words_lst', type=str, nargs='+', default=["５/m", "虽然/c", "天/q"], help='if not random, get the words analysed')
    args = parser.parse_args()

    dataset = torch.load("datasets/save_dataset_9_1024_rnn.pth")
    top_words = dataset.top_words

    words_lst = []
    if args.random:
        for i in range(20):
            word = random.choice(list(top_words.keys()))
            words_lst.append(word)
        print("words list:", words_lst)
    else:
        words_lst = args.words_lst
        print("words list:", words_lst)

    lookup_table_dict = {
        "FNN": load_lookup_table("FNN"),
        "RNN": load_lookup_table("RNN"),
        "LSTM": load_lookup_table("LSTM")
    }

    for word in words_lst:
        print("----------------------------------")
        print(word)
        for model_name in ["FNN", "RNN", "LSTM"]:
            lookup_table = lookup_table_dict[model_name]
            print("top 10 similar words in " + model_name + "：")
            orders, similarity = get_similar_wordidx(lookup_table, top_words[word])
            for idx, order in enumerate(orders[:10]):
                print("{0} :".format(idx), get_word(top_words, order), similarity[order])


    with open("result.txt", "w") as f:
        for word in words_lst:
            f.write("----------------------------------\n")
            f.write(word + "\n")
            for model_name in ["FNN", "RNN", "LSTM"]:
                lookup_table = lookup_table_dict[model_name]
                f.write("top 10 similar words in " + model_name + "：\n")
                orders, similarity = get_similar_wordidx(lookup_table, top_words[word])
                for idx, order in enumerate(orders[:10]):
                    f.write("{0} : ".format(idx) + get_word(top_words, order) + ' ' + str(similarity[order]) + "\n")