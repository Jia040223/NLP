import sklearn_crfsuite
import os
from joblib import dump,load


class CRFsModel:
    def __init__(self):
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def transfer_dataset(self, file_path, encoding="gbk"):
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        bmes_data = []
        bmes_label = []

        for line in lines:
            words = line.split()
            words = [word.split('/')[0] for word in words]
            sentence = "".join(words)
            bmes_data.append(sentence)

            line_label = []
            for word in words:
                word = word.split('/')[0]
                if len(word) == 1:
                    line_label.append('S')
                else:
                    line_label.append('B')
                    for char in word[1:-1]:
                        line_label.append('M')
                    line_label.append('E')
            bmes_label.append(line_label)

        return bmes_data, bmes_label

    def word2features(self, sent, i):
        word = sent[i][0]
        # 构造特征字典
        features = {
            'bias': 1.0,
            'word': word,
            'word.isdigit()': word.isdigit(),
        }
        # 该字的前一个字
        if i > 0:
            word1 = sent[i - 1][0]
            words = word1 + word
            features.update({
                '-1:word': word1,
                '-1:words': words,
                '-1:word.isdigit()': word1.isdigit(),
            })
        else:
            # 添加开头的标识 BOS(begin of sentence)
            features['BOS'] = True
        # 该字的前两个字
        if i > 1:
            word2 = sent[i - 2][0]
            word1 = sent[i - 1][0]
            words = word1 + word2 + word
            features.update({
                '-2:word': word2,
                '-2:words': words,
                '-3:word.isdigit()': word2.isdigit(),
            })
        # 该字的前三个字
        if i > 2:
            word3 = sent[i - 3][0]
            word2 = sent[i - 2][0]
            word1 = sent[i - 1][0]
            words = word1 + word2 + word3 + word
            features.update({
                '-3:word': word3,
                '-3:words': words,
                '-3:word.isdigit()': word3.isdigit(),
            })
        # 该字的后一个字
        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            words = word1 + word
            features.update({
                '+1:word': word1,
                '+1:words': words,
                '+1:word.isdigit()': word1.isdigit(),
            })
        else:
            # 句子的结尾添加对应的标识end of sentence
            features['EOS'] = True
        # 该字的后两个字
        if i < len(sent) - 2:
            word2 = sent[i + 2][0]
            word1 = sent[i + 1][0]
            words = word + word1 + word2
            features.update({
                '+2:word': word2,
                '+2:words': words,
                '+2:word.isdigit()': word2.isdigit(),
            })
        # 该字的后三个字
        if i < len(sent) - 3:
            word3 = sent[i + 3][0]
            word2 = sent[i + 2][0]
            word1 = sent[i + 1][0]
            words = word + word1 + word2 + word3
            features.update({
                '+3:word': word3,
                '+3:words': words,
                '+3:word.isdigit()': word3.isdigit(),
            })
        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(sent):
        return sent

    def load_data(self, file_path, encoding="gbk"):
        bmes_data, bmes_label = self.transfer_dataset(file_path, encoding)

        self.train_data = [self.sent2features(s) for s in bmes_data]
        self.train_label = bmes_label

    def train(self):
        self.crf.fit(self.train_data, self.train_label)

    def segment(self, texts):
        if not isinstance(texts, list):
            texts = [texts]

        datas = [self.sent2features(text) for text in texts]

        # 使用模型进行预测
        labels = self.crf.predict(datas)

        # 根据预测的标签进行分词
        predict_result = []
        for text, label in zip(texts, labels):
            words = []
            word = ''
            for char, label in zip(text, label):
                if label in ['/B', '/S']:
                    if word:
                        words.append(word)
                    word = char
                else:
                    word += char
            if word:
                words.append(word)

            predict_result.append(words)

        return predict_result

    def save_model(self, file_path):
        dump(self.crf, file_path)

    def load_model(self, file_path):
        self.crf = load(file_path)


if __name__ == "__main__":
    crf = CRFsModel()
    if os.path.exists("./checkpoint/crf_model.joblib"):
        print(os.path.abspath("./checkpoint/crf_model.joblib"))
        crf.load_model("./checkpoint/crf_model.joblib")
    else:
        crf.load_data("./ChineseCorpus199801.txt")
        crf.train()
        crf.save_model("./checkpoint/crf_model.joblib")

    text = "差不多得了，屁大点事都要拐上原神，原神一没招你惹你，二没干伤天害理的事情，到底怎么你了让你一直无脑抹黑，米哈游每天费尽心思的文化输出弘扬中国文化，你这种喷子只会在网上敲键盘诋毁良心公司，中国游戏的未来就是被你这种人毁掉的。"
    words = crf.segment(text)
    print(words)