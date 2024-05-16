import sklearn_crfsuite
import os
from joblib import dump, load


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
            word = []
            for token, label in zip(text, label):
                if label == '/B':
                    if word:
                        words.append(''.join(word))
                        word = []
                    word.append(token)
                elif label == '/M':
                    word.append(token)
                elif label == '/E':
                    word.append(token)
                    words.append(''.join(word))
                    word = []
                else:
                    if word:
                        words.append(''.join(word))
                        word = []
                    words.append(token)

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

    texts = ["党中央和国务院高度重视高校毕业生等青年就业创业工作。要深入学习贯彻总书记的重要指示精神，更加突出就业优先导向，千方百计促进高校毕业生就业，确保青年就业形势总体稳定。",
             "好久不见！今天天气真好，早饭准备吃什么呀？",
             "我特别喜欢去北京的天安门和颐和园进行游玩",
             "中国人为了实现自己的梦想",
             "《原神》收入大涨，腾讯、网易、米哈游位列中国手游发行商全球收入前三"]
    for text in texts:
        words = crf.segment(text)
        print("/".join(words[0]))
