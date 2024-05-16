from collections import Counter
import os
from joblib import dump,load

class BiMM:
    def __init__(self, load=False, file_name=None, dictionary=None, encoding="gbk"):
        if dictionary is not None:
            self.dictionary = dictionary
        elif file_name is not None and not load:
            self.dictionary = self.create_dictionary(file_name, encoding=encoding)
        else:
            self.load_dictionary(file_name)

        self.max_len = max(len(word) for word in self.dictionary)

    def create_dictionary(self, data_path, encoding):
        with open(data_path, 'r', encoding=encoding) as f:
            words = f.read().split()

        word_count = Counter(words)

        word_dict = set()
        for word, count in word_count.most_common():
            word = word.split('/')[0]  # 提取词语
            if word.startswith('[') and len(word) > 1:  # 如果词语以"["开头，并且长度大于1
                word = word[1:]  # 去除左方括号
            word_dict.add(word)

        return word_dict

    def forward_segment(self, text):
        text = text.strip()
        words_length = len(text)  # 统计序列长度
        # 存储切分好的词语
        cut_word_list = []
        while words_length > 0:
            max_cut_length = min(self.max_len, words_length)
            sub_text = text[0: max_cut_length]
            while max_cut_length > 0:
                if sub_text in self.dictionary:
                    cut_word_list.append(sub_text)
                    break
                elif max_cut_length == 1:
                    cut_word_list.append(sub_text)
                    break
                else:
                    max_cut_length = max_cut_length - 1
                    sub_text = sub_text[0:max_cut_length]
            text = text[max_cut_length:]
            words_length = words_length - max_cut_length
        return cut_word_list

    def backward_segment(self, text):
        text = text.strip()
        words_length = len(text)  # 统计序列长度
        cut_word_list = []  # 存储切分出来的词语
        # 判断是否需要继续切词
        while words_length > 0:
            max_cut_length = min(self.max_len, words_length)
            sub_text = text[-max_cut_length:]
            while max_cut_length > 0:
                if sub_text in self.dictionary:
                    cut_word_list.append(sub_text)
                    break
                elif max_cut_length == 1:
                    cut_word_list.append(sub_text)
                    break
                else:
                    max_cut_length = max_cut_length - 1
                    sub_text = sub_text[-max_cut_length:]
            text = text[0:-max_cut_length]
            words_length = words_length - max_cut_length

        cut_word_list.reverse()
        return cut_word_list

    def segment(self, text):
        bmm_word_list = self.backward_segment(text)
        fmm_word_list = self.forward_segment(text)
        bmm_word_list_size = len(bmm_word_list)
        fmm_word_list_size = len(fmm_word_list)
        if bmm_word_list_size != fmm_word_list_size:
            if bmm_word_list_size < fmm_word_list_size:
                return bmm_word_list
            else:
                return fmm_word_list
        else:
            FSingle = 0
            BSingle = 0
            isSame = True
            for i in range(len(fmm_word_list)):
                if fmm_word_list[i] not in bmm_word_list:
                    isSame = False
                if len(fmm_word_list[i]) == 1:
                    FSingle = FSingle + 1
                if len(bmm_word_list[i]) == 1:
                    BSingle = BSingle + 1
            if isSame:
                return fmm_word_list
            elif BSingle > FSingle:
                return fmm_word_list
            else:
                return bmm_word_list

    def save_dictionary(self, file_path):
        dump(self.dictionary, file_path)

    def load_dictionary(self, file_path):
        self.dictionary = load(file_path)

if __name__ == "__main__":
    data_path = "./ChineseCorpus199801.txt"
    dic_path = "./checkpoint/biMM_dic.joblib"

    if os.path.exists(dic_path):
        bimm = BiMM(load=True, file_name=dic_path)
    else:
        bimm = BiMM(file_name=data_path)
        bimm.save_dictionary(dic_path)

    texts = ["党中央和国务院高度重视高校毕业生等青年就业创业工作。要深入学习贯彻总书记的重要指示精神，更加突出就业优先导向，千方百计促进高校毕业生就业，确保青年就业形势总体稳定。",
             "好久不见！今天天气真好，早饭准备吃什么呀？",
             "我特别喜欢去北京的天安门和颐和园进行游玩",
             "中国人为了实现自己的梦想",
             "《原神》收入大涨，腾讯、网易、米哈游位列中国手游发行商全球收入前三"]
    for text in texts:
        words = bimm.segment(text)
        print("/".join(words))
