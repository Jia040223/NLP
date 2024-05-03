from collections import Counter

def data_process(filename):
    # 读取文本文件
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 用于存放单词的列表
    words = []

    # 遍历每行文本
    for line in lines:
        # 删除每行的前缀
        words.extend(line.split())

    # 统计单词出现的次数
    word_counts = Counter(words)

    # print word_counts to file according to the frequency of occurrence
    with open('words_count.txt', 'w') as f:
        for word, count in word_counts.most_common():
            f.write(word + ' ' + str(count) + '\n')

if __name__ == "__main__":
    data_process("ChineseCorpus199801.txt")