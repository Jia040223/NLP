
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端

def read_results(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    results = {
        "train_losses": [],
        "train_precisions": [],
        "train_recalls": [],
        "train_f1s": [],
        "test_losses": [],
        "test_precisions": [],
        "test_recalls": [],
        "test_f1s": []
    }

    key = None
    for line in lines:
        line = line.strip()
        if "Train Losses:" in line:
            key = "train_losses"
        elif "Train Precisions:" in line:
            key = "train_precisions"
        elif "Train Recalls:" in line:
            key = "train_recalls"
        elif "Train F1 Scores:" in line:
            key = "train_f1s"
        elif "Test Losses:" in line:
            key = "test_losses"
        elif "Test Precisions:" in line:
            key = "test_precisions"
        elif "Test Recalls:" in line:
            key = "test_recalls"
        elif "Test F1 Scores:" in line:
            key = "test_f1s"
        elif key is not None and line:  # Check if line is not empty
            try:
                results[key].append(float(line))
            except ValueError:
                print(f"Skipping invalid line: {line}")

    return results


def plot_results(results):
    epochs = range(1, len(results["train_losses"]) + 1)

    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, results["train_losses"], 'b', label='Train Loss')
    plt.plot(epochs, results["test_losses"], 'r', label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(epochs, results["train_precisions"], 'b', label='Train Precision')
    plt.plot(epochs, results["test_precisions"], 'r', label='Test Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, results["train_recalls"], 'b', label='Train Recall')
    plt.plot(epochs, results["test_recalls"], 'r', label='Test Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Plot F1 scores
    plt.subplot(2, 2, 4)
    plt.plot(epochs, results["train_f1s"], 'b', label='Train F1')
    plt.plot(epochs, results["test_f1s"], 'r', label='Test F1')
    plt.title('F1 Scores')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def record_training(file_name, train_losses, train_precisions, train_recalls, train_f1s,
            test_losses, test_precisions, test_recalls, test_f1s):
    with open(file_name, 'w') as file:
        file.write("Train Losses:\n")
        for loss in train_losses:
            file.write(str(loss) + '\n')

        file.write("\nTrain Precisions:\n")
        for precision in train_precisions:
            file.write(str(precision) + '\n')

        file.write("\nTrain Recalls:\n")
        for recall in train_recalls:
            file.write(str(recall) + '\n')

        file.write("\nTrain F1 Scores:\n")
        for f1 in train_f1s:
            file.write(str(f1) + '\n')

        file.write("\nTest Losses:\n")
        for loss in test_losses:
            file.write(str(loss) + '\n')

        file.write("\nTest Precisions:\n")
        for precision in test_precisions:
            file.write(str(precision) + '\n')

        file.write("\nTest Recalls:\n")
        for recall in test_recalls:
            file.write(str(recall) + '\n')

        file.write("\nTest F1 Scores:\n")
        for f1 in test_f1s:
            file.write(str(f1) + '\n')


if __name__ == "__main__":
    filename = 'training_results_bert.txt'
    results = read_results(filename)
    plot_results(results)
