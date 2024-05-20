import argparse
import yaml
import torch
import torch.optim as optim
from torch import nn
import numpy as np
from transformers import BertForTokenClassification,BertConfig
from dataloader import get_loaders
from dataset import AutoTokenizer
import os
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score
from record import record_training


class Model_Pipline():
    def __init__(self, config, device="cuda", load_path=None, row_filename="ChineseCorpus199801.txt"):
        self.filename = row_filename

        cuda_availdabe = torch.cuda.is_available()
        if cuda_availdabe and device != "cpu":
            print('Initializing model on GPU')
        else:
            print('Initializing model on CPU')

        # 超参数
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.label2id = config["label2id"]

        bert_config = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
        self.model = BertForTokenClassification(bert_config)

        if cuda_availdabe and device != "cpu":
            self.model.cuda()

        if load_path is not None:
            self.model.load_state_dict(load_path)

    def train_or_eval(self, loss_function, dataloader, optimizer=None, train=False):
        losses, precisions, recalls, f1s = [], [], [], []

        assert not train or optimizer != None
        if train:
            self.model.train()
        else:
            self.model.eval()

        num_batches = len(dataloader)

        for data in dataloader:
            if train:
                optimizer.zero_grad()

            input_ids, token_type_ids, labels, attention_mask = [d.cuda() for d in data[:]] if torch.cuda.is_available() else data[:]

            final_out = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            logits = final_out.logits
            loss = loss_function(logits.view(-1, self.model.num_labels), labels.view(-1))

            losses.append(loss.item())

            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

            predictions = torch.argmax(logits, dim=-1)

            # 将标签和预测结果添加到列表中
            true_labels = labels.tolist()
            pred_labels = predictions.tolist()

            # 计算各种评估指标
            bmseo_to_iobes = {'B': 'B', 'M': 'I', 'E': 'E', 'S': 'S', '0': 'O'}
            inverse_label_map = {v: k for k, v in self.label2id.items()}
            true_labels_str = [[inverse_label_map[label] for label in sequence] for sequence in true_labels]
            pred_labels_str = [[inverse_label_map[label] for label in sequence] for sequence in pred_labels]

            true_labels_iobes = [[bmseo_to_iobes[label] for label in sequence] for sequence in true_labels_str]
            pred_labels_iobes = [[bmseo_to_iobes[label] for label in sequence] for sequence in pred_labels_str]

            precision = precision_score(true_labels_iobes, pred_labels_iobes)
            recall = recall_score(true_labels_iobes, pred_labels_iobes)
            f1 = f1_score(true_labels_iobes, pred_labels_iobes)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        avg_loss = round(np.sum(losses), 4)
        avg_loss /= num_batches
        avg_precision = round(np.sum(precisions), 4)
        avg_precision /= num_batches
        avg_recalls = round(np.sum(recalls), 4)
        avg_recalls /= num_batches
        avg_f1 = round(np.sum(f1s), 4)
        avg_f1 /= num_batches

        return avg_loss, avg_precision, avg_recalls, avg_f1

    def train(self, optimizer=None, loss_function=None):
        if optimizer is None:
            '''
            weight_decay = 0.01
            no_decay = ["bias", "norm"]

            parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            '''

            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        train_loader, _, test_loader = get_loaders(label2id=self.label2id, batch_size=self.batch_size, valid=0.0,
                                                   train=0.8, filename=self.filename)

        cumulative_epoch = self.load_model()

        train_losses, train_precisions, train_recalls, train_f1s = [], [], [], []
        test_losses, test_precisions, test_recalls, test_f1s = [], [], [], []
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for _ in pbar:
            train_loss, train_precision, train_recall, train_f1 = self.train_or_eval(loss_function, train_loader, optimizer, True)
            test_loss, test_precision, test_recall, test_f1 = self.train_or_eval(loss_function, test_loader)

            # 将训练期间的性能指标添加到列表中
            train_losses.append(train_loss)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)
            train_f1s.append(train_f1)

            # 将测试期间的性能指标添加到列表中
            test_losses.append(test_loss)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)
            test_f1s.append(test_f1)

            # 更新进度条的显示信息
            pbar.set_postfix(train_loss=train_loss, train_precision=train_precision, train_recall=train_recall, train_f1=train_f1,
                             test_loss=test_loss, test_precision=test_precision, test_recall=test_recall, test_f1=test_f1)
            '''
            print("epoch{0} : train_loss={1}, train_precision={2}, train_recall={3}, train_f1={4}"
                  .format(cumulative_epoch, train_loss, train_precision, train_recall, train_f1))
            print("epoch{0} : test_loss={1}, test_precision={2}, test_recall={3}, test_f1={4}"
                  .format(cumulative_epoch, test_loss, test_precision, test_recall, test_f1))
            '''

            cumulative_epoch += 1
            if cumulative_epoch % 1 == 0:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(self.model.state_dict(), "./checkpoint/Bert-" + str(cumulative_epoch) + ".pth")

        pbar.close()

        return train_losses, train_precisions, train_recalls, train_f1s, \
            test_losses, test_precisions, test_recalls, test_f1s

    def test(self, dataloader=None, loss_function=None):
        self.load_model()

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
        if dataloader is None:
            _, _, dataloader = get_loaders(label2id=self.label2id, batch_size=self.batch_size, valid=0.0,
                                                       train=0.0, filename=self.filename)

        test_loss, test_precision, test_recall, test_f1 = self.train_or_eval(loss_function, dataloader)

        return test_loss, test_precision, test_recall, test_f1

    def segment(self, text):
        id2label = {v: k for k, v in self.label2id.items()}

        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        raw_texts = [text]
        encoded_inputs = tokenizer(raw_texts, max_length=512)
        input_ids = torch.tensor(encoded_inputs['input_ids']).cuda()
        token_type_ids = torch.tensor(encoded_inputs['token_type_ids']).cuda()

        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).tolist()
        predictions = [[id2label[label_id] for label_id in raw[1:-1]] for raw in predictions]

        results = []
        for text, labels in zip(raw_texts, predictions):
            words = []
            word = []
            for token, label in zip(text, labels):
                if label == 'B':
                    if word:
                        words.append(''.join(word))
                        word = []
                    word.append(token)
                elif label == 'M':
                    word.append(token)
                elif label == 'E':
                    word.append(token)
                    words.append(''.join(word))
                    word = []
                else:
                    if word:
                        words.append(''.join(word))
                        word = []
                    words.append(token)

            results.append(words)

        return results

    def load_model(self, load_path=None):
        return_id = 0
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
        else:
            pth_list = os.listdir("checkpoint")
            latest_pth = None
            for pth in pth_list:
                if pth.endswith(".pth"):
                    if latest_pth is None:
                        latest_pth = pth
                    else:
                        current_id = int(pth.split("-")[-1].split(".")[0])
                        latest_id = int(latest_pth.split("-")[-1].split(".")[0])
                        if current_id > latest_id:
                            latest_pth = pth

            if latest_pth is not None:
                print("load model from checkpoint/" + latest_pth)
                self.model.load_state_dict(torch.load("checkpoint/" + latest_pth))
                return_id = int(latest_pth.split("-")[-1].split(".")[0])

        return return_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The setting of bert models')
    parser.add_argument('--train', type=str, default=True, help='train or not')
    parser.add_argument('--config', type=str, default="./config/base.yaml", help='config file')
    parser.add_argument('--device', type=str, default="cuda", help='the device to train or test the model')
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    Model = Model_Pipline(config, device)

    if args.train is True:
        train_loss, train_precision, train_recall, train_f1, \
        test_loss, test_precision, test_recall, test_f1 = Model.train()

        filename = 'training_results_bert.txt'
        record_training(filename, train_loss, train_precision, train_recall, train_f1,
                        test_loss, test_precision, test_recall, test_f1)
    else:
        #test_losses, test_precisions, test_recalls, test_f1s = Model.test()
        #print("test loss :{0}, test_precison:{1}, test_recalls:{2}, test_f1s:{3}".formate(test_losses, test_precisions, test_recalls, test_f1s)
        Model.load_model()
        texts = ["党中央和国务院高度重视高校毕业生等青年就业创业工作。要深入学习贯彻总书记的重要指示精神，更加突出就业优先导向，千方百计促进高校毕业生就业，确保青年就业形势总体稳定。",
                 "好久不见！今天天气真好，早饭准备吃什么呀？",
                 "我特别喜欢去北京的天安门和颐和园进行游玩",
                 "中国人为了实现自己的梦想",
                 "《原神》收入大涨，腾讯、网易、米哈游位列中国手游发行商全球收入前三"]

        for text in texts:
            results = Model.segment(text)
            print("/".join(results[0]))
