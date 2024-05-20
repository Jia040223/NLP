import re
from collections import defaultdict
from bi_MM import BiMM
from CRF import CRFsModel
import argparse
import yaml
import torch
from bert import Model_Pipline

class BPE():
    def __init__(self, model=None, raw_file_path=None, use_model=False, num_merges=1000000):
        self.model = model
        self.vocab = None

        if raw_file_path is not None:
            with open(raw_file_path, "r") as f:
                lines = f.readlines()

            pattern = re.compile(r'\d{8}-\d{2}-\d{3}-\d{3}/\w')

            self.texts = []
            for line in lines:
                line = pattern.sub('', line)
                words = line.split()
                words = [word.split('/')[0] for word in words]
                if use_model:
                    sentence = "".join(words)
                    self.texts.append(self.model.segment(sentence))
                else:
                    self.texts.append(words)

            self.get_vocab(self.texts)

        self.bpe_vocab(num_merges)
        self.get_tokens()

        assert self.vocab is not None and self.model is not None

    def get_vocab(self, texts: list) -> None:
        # 基础分词，获取词表
        self.vocab = defaultdict(int)
        for text in texts:
            for word in text:
                # 分割成字符并在结尾加上终止符
                word = ' '.join(list(word)) + ' </w>'
                self.vocab[word] += 1

    def get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq

        return pairs

    def merge_vocab(self, pair):
        bigram = re.escape(' '.join(pair)) #转义空格
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') #只匹配单词的开头或者空格后的字符对
        new_vocab = defaultdict(int)
        for word in self.vocab:
            w_out = pattern.sub(''.join(pair), word)
            new_vocab[w_out] += self.vocab[word]

        return new_vocab

    def bpe_vocab(self, num_merges):
        if self.vocab is None:
            self.get_vocab(self.texts)

        for _ in range(num_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            #if pairs[best] == 1:
                #break
            self.vocab = self.merge_vocab(best)

    def get_tokens(self):
        self.tokens = defaultdict(int)
        for word, freq in self.vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                self.tokens[token] += freq
        return self.tokens

    def segment(self, sentences):
        dic = []
        for key in self.tokens.keys():
            key.replace("\w", "")
            if key not in dic:
                dic.append(key)

        results = []
        for sentence in sentences:
            try:
                sentence = self.model.segment(sentence)[0]
                sentence = "".join(sentence)
            except:
                sentence = self.model.segment(sentence)
                sentence = "".join(sentence)

            segment_tools = BiMM(dictionary=dic)
            results.append(segment_tools.segment(sentence))

        return results

    def segment_w(self, sentences):
        results = []
        for sentence in sentences:
            try:
                sentence = [text+'</w>' for text in self.model.segment(sentence)[0]]
                sentence = "".join(sentence)
            except:
                sentence = [text+'</w>' for text in self.model.segment(sentence)]
                sentence = "".join(sentence)
            segment_tools = BiMM(dictionary=self.tokens.keys())
            results.append(segment_tools.segment(sentence))

        return results

    def bpe_decode(self, sentences):
        results = []
        for sentence in sentences:
            if isinstance(sentence, list):
                sentence = "".join(sentence)

            decoded_text = []
            word = ""
            for char in sentence:
                if char == "<":  # 开始一个新的词或子词
                    if word:  # 如果之前的词不为空，将其加入解码文本中
                        decoded_text.append(word)
                    word = ""
                elif char == ">" or char == "w" or char == "/":
                    continue
                else:
                    word += char  # 继续构建当前词
            if word:  # 如果最后一个词不为空，将其加入解码文本中
                decoded_text.append(word)

            results.append(decoded_text)

        return decoded_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The setting of BRE')
    parser.add_argument('--model', type=str, default="CRF", help='segment model: (BI_MM, CRF or BERT)')
    parser.add_argument('--config', type=str, default="./config/base.yaml", help='config file')
    parser.add_argument('--device', type=str, default="cuda", help='the device to train or test the model')
    args = parser.parse_args()

    if args.model == "CRF":
        crf = CRFsModel()
        crf.load_model("./checkpoint/crf_model.joblib")
        bpe = BPE(model=crf, raw_file_path="ChineseCorpus199801.txt", num_merges=100)
        text = ["党中央和国务院高度重视高校毕业生等青年就业创业工作。要深入学习贯彻总书记的重要指示精神，更加突出就业优先导向，千方百计促进高校毕业生就业，确保青年就业形势总体稳定。"]
        print(bpe.segment(text))
        text_w = bpe.segment_w(text)
        print(text_w)
        print(bpe.bpe_decode(text_w))
    elif args.model == "BERT":
        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"
        device = args.device

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        bert = Model_Pipline(config, device)
        bert.load_model()

        bpe = BPE(model=bert, raw_file_path="ChineseCorpus199801.txt", num_merges=100)

        text = ["党中央和国务院高度重视高校毕业生等青年就业创业工作。要深入学习贯彻总书记的重要指示精神，更加突出就业优先导向，千方百计促进高校毕业生就业，确保青年就业形势总体稳定。"]
        print(bpe.segment(text))
        text_w = bpe.segment_w(text)
        print(text_w)
        print(bpe.bpe_decode(text_w))
    else:
        dic_path = "./checkpoint/biMM_dic.joblib"
        bimm = BiMM(load=True, file_name=dic_path)

        bpe = BPE(model=bimm, raw_file_path="ChineseCorpus199801.txt", num_merges=100)

        text = ["党中央和国务院高度重视高校毕业生等青年就业创业工作。要深入学习贯彻总书记的重要指示精神，更加突出就业优先导向，千方百计促进高校毕业生就业，确保青年就业形势总体稳定。"]
        print(bpe.segment(text))
        text_w = bpe.segment_w(text)
        print(text_w)
        print(bpe.bpe_decode(text_w))