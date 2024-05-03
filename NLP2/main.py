import argparse
import yaml
import torch
import torch.optim as optim
from torch import nn
import numpy as np
from model.models import LSTM, RNN, FNN
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

        # 维度
        self.vocab_size = config["vocab_size"]
        self.embedding_dim = config["embedding_dim"]
        self.hidden_size = config["hidden_size"]
        self.n = config["n"]
        self.layer = config["layer"]

        if config["model"] == "FNN":
            self.dataset_type = "n_gram"
        else:
            self.dataset_type = "rnn"

        if load_path is not None:
            if config["model"] == "FNN":
                self.model = FNN(self.vocab_size, self.n, self.embedding_dim, self.hidden_size, self.layer).load_state_dict(load_path)
                self.type = "FNN"
            elif config["model"] == "RNN":
                self.model = RNN(self.vocab_size, self.embedding_dim, self.hidden_size, self.layer).load_state_dict(load_path)
                self.type = "RNN"
            elif config["model"] == "LSTM":
                self.model = LSTM(self.vocab_size, self.embedding_dim, self.hidden_size, self.layer).load_state_dict(load_path)
                self.type = "LSTM"
            else:
                raise Exception("Unknown model type")
        else:
            if config["model"] == "FNN":
                self.model = FNN(self.vocab_size, self.n, self.embedding_dim, self.hidden_size, self.layer)
                self.type = "FNN"
            elif config["model"] == "RNN":
                self.model = RNN(self.vocab_size, self.embedding_dim, self.hidden_size, self.layer)
                self.type = "RNN"
            elif config["model"] == "LSTM":
                self.model = LSTM(self.vocab_size, self.embedding_dim, self.hidden_size, self.layer)
                self.type = "LSTM"
            else:
                raise Exception("Unknown model type")

        if cuda_availdabe and device != "cpu":
            self.model.cuda()

    def train_or_eval(self, loss_function, dataloader, optimizer=None, train=False):
        losses = []

        assert not train or optimizer != None
        if train:
            self.model.train()
        else:
            self.model.eval()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        correct_rate = 0
        for data in dataloader:
            if train:
                optimizer.zero_grad()

            input_tensor, output_tensor = [d.cuda() for d in data[:]] if torch.cuda.is_available() else data[:]

            final_out = self.model(input_tensor)
            loss = loss_function(final_out.reshape(-1, final_out.shape[-1]), output_tensor.flatten())
            correct_rate += (final_out.argmax(-1) == output_tensor).type(torch.float).sum().item() * \
                            final_out.shape[0] / np.prod(final_out.shape[:-1])
            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

        avg_loss = round(np.sum(losses), 4)
        avg_loss /= num_batches
        correct_rate /= size

        return avg_loss, correct_rate

    def train(self, optimizer=None, loss_function=None):
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()

        train_loader, _, test_loader = get_loaders(batch_size=self.batch_size, valid=0.0, train=0.8, filename=self.filename,
                                                   n_size=self.n, vocab_size=self.vocab_size, type=self.dataset_type)

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
            train_loss, train_correct = self.train_or_eval(loss_function, train_loader, optimizer, True)
            test_loss, test_correct = self.train_or_eval(loss_function, test_loader)

            train_losses.append(train_loss)
            train_accuracy.append(train_correct)
            test_losses.append(test_loss)
            test_accuracy.append(test_correct)

            # 更新进度条的显示信息
            pbar.set_postfix(train_loss=train_loss, train_accuracy=train_correct,
                             test_loss=test_loss, test_accuracy=test_correct)
            #print('epoch: {}, train_loss: {}, train_correct_rate:{}, test_loss: {}, test_correct_rate:{}'.format(cumulative_epoch, train_loss, train_correct, test_loss, test_correct))

            cumulative_epoch += 1
            if cumulative_epoch % 5 == 0:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(self.model.state_dict(), f'./checkpoint/{config["model"]}-' + str(cumulative_epoch) + '.pth')

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
            _, _, dataloader = get_loaders(batch_size=self.batch_size, valid=0.0, train=0.0, filename=self.filename,
                                           n_size=self.n, vocab_size=self.vocab_size, type=self.dataset_type)

        test_loss, test_correct = self.train_or_eval(loss_function, dataloader)

        print(f"Test: \n Accuracy: {(100 * test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return test_correct, test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The setting of different models')
    parser.add_argument('--model', type=str, default='LSTM', help='model name(FNN, RNN, LSTM)')
    parser.add_argument('--train', type=str, default=False, help='train or not')
    parser.add_argument('--config', type=str, default="./config/base.yaml", help='config file')
    parser.add_argument('--device', type=str, default="cuda", help='the device to train or test the model')
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)

    if args.model == "FNN":
        config = configs["FNN"]
        config["model"] = "FNN"
    elif args.model == "RNN":
        config = configs["RNN"]
        config["model"] = "RNN"
    elif args.model == "LSTM":
        config = configs["LSTM"]
        config["model"] = "LSTM"

    #print(config)
    Model = Model_Pipline(config, device)

    if args.train is True:
        train_losses, train_accuracy, test_losses, test_accuracy = Model.train()
        # 将变量写入文件
        with open('training_results_LSTM.txt', 'w') as file:
            file.write("Train Losses:\n")
            for loss in train_losses:
                file.write(str(loss) + '\n')

            file.write("\nTrain Accuracy:\n")
            for accuracy in train_accuracy:
                file.write(str(accuracy) + '\n')

            file.write("\nTest Losses:\n")
            for loss in test_losses:
                file.write(str(loss) + '\n')

            file.write("\nTest Accuracy:\n")
            for accuracy in test_accuracy:
                file.write(str(accuracy) + '\n')

        plot_metrics(train_losses, train_accuracy, test_losses, test_accuracy)
    else:
        _, _ = Model.test()

