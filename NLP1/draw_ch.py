import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 读取数据文件
with open('entropy_ch.txt', 'r') as file:
    data = file.readlines()

if __name__ == "__main__":
    # 初始化空列表以存储 x 和 y 坐标数据
    x_values = []
    y_values = []

    # 将数据拆分为 x 和 y 坐标
    for line in data:
        parts = line.strip().split(':')
        x_values.append(float(parts[0]))
        y_values.append(float(parts[1]))

    # 绘制曲线图
    plt.plot(x_values, y_values)
    plt.xlabel('File Size (MB)')
    plt.ylabel('Entropy')
    plt.title('Entropy vs. File Size')
    plt.show()
