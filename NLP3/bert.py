import argparse
import yaml
import torch
import torch.optim as optim
from torch import nn
import numpy as np
from transformers import BertForTokenClassification,BertConfig
from dataloader import get_loaders
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use('TkAgg')  # 使用TkAgg后端

def plot_metrics(train_losses, train_accuracy, test_losses, test_accuracy):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # 绘制训练和测试损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

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

    def train_or_eval(self, loss_function, dataloader, optimizer=None, train=False):
        losses = []

        assert not train or optimizer != None
        if train:
            self.model.train()
        else:
            self.model.eval()

        num_batches = len(dataloader)

        for data in dataloader:
            if train:
                optimizer.zero_grad()

            input_ids, token_type_ids, labels = [d.cuda() for d in data[:]] if torch.cuda.is_available() else data[:]

            final_out = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
            logits = final_out.logits
            loss = loss_function(logits.view(-1, self.model.num_labels), labels.view(-1))

            losses.append(loss.item())

            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

        avg_loss = round(np.sum(losses), 4)
        avg_loss /= num_batches

        return avg_loss

    def train(self, optimizer=None, loss_function=None):
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        train_loader, _, test_loader = get_loaders(label2id=self.label2id, batch_size=self.batch_size, valid=0.0,
                                                   train=0.8, filename=self.filename)

        pth_list = os.listdir("checkpoint")
        latest_pth = None
        cumulative_epoch = 0
        for pth in pth_list:
            if pth.endswith(".pth") and pth.startswith(self.type):
                if latest_pth is None:
                    latest_pth = pth
                else:
                    current_id = int(pth.split("-")[-1].split(".")[0])
                    latest_id = int(latest_pth.split("-")[-1].split(".")[0])
                    if current_id > latest_id:
                        latest_pth = pth
                        cumulative_epoch = current_id

        if latest_pth is not None:
            print("load model from checkpoint/" + latest_pth)
            self.model.load_state_dict(torch.load("checkpoint/" + latest_pth))

        train_losses, train_accuracy, test_losses, test_accuracy = [], [], [], []
        pbar = tqdm(range(self.epochs), desc="Training Progress")
        for _ in pbar:
            train_loss = self.train_or_eval(loss_function, train_loader, optimizer, True)
            test_loss = self.train_or_eval(loss_function, test_loader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # 更新进度条的显示信息
            pbar.set_postfix(train_loss=train_loss,
                             test_loss=test_loss)
            #print('epoch: {}, train_loss: {}, train_correct_rate:{}, test_loss: {}, test_correct_rate:{}'.format(cumulative_epoch, train_loss, train_correct, test_loss, test_correct))

            cumulative_epoch += 1
            if cumulative_epoch % 1 == 0:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(self.model.state_dict(), "./checkpoint/Bert-" + str(cumulative_epoch) + ".pth")

        pbar.close()

        return train_losses, train_accuracy, test_losses, test_accuracy

    def test(self, dataloader=None, loss_function=None):
        pth_list = os.listdir("checkpoint")
        latest_pth = None
        for pth in pth_list:
            if pth.endswith(".pth") and pth.startswith(self.type):
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

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
        if dataloader is None:
            _, _, test_loader = get_loaders(label2id=self.label2id, batch_size=self.batch_size, valid=0.0,
                                                       train=0.8, filename=self.filename)

        test_loss = self.train_or_eval(loss_function, dataloader)

        return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The setting of different models')
    parser.add_argument('--train', type=str, default=True, help='train or not')
    parser.add_argument('--config', type=str, default="./config/base.yaml", help='config file')
    parser.add_argument('--device', type=str, default="cuda", help='the device to train or test the model')
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    #print(config)
    Model = Model_Pipline(config, device)

    if args.train is True:
        train_losses, test_losses = Model.train()
        # 将变量写入文件
        with open('training_results_LSTM.txt', 'w') as file:
            file.write("Train Losses:\n")
            for loss in train_losses:
                file.write(str(loss) + '\n')

            file.write("\nTest Losses:\n")
            for loss in test_losses:
                file.write(str(loss) + '\n')

        #plot_metrics(train_losses, test_losses)
    else:
        _, _ = Model.test()
