import re
import math


def calculate_entropy(rate, content_len):
    # 计算汉字熵
    entropy = 0
    for key, value in rate.items():
        entropy += value / content_len * math.log(content_len / value) / math.log(2)

    return entropy

def record_entropy(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.readlines()
        content_len = 0

        characers = []  # 存放不同字的总数
        rate = {}  # 存放每个字出现的频率

        record_num = 1

        # 依次迭代所有行
        for line in content:
            # 去除空格
            line = line.strip()
            # 如果是空行，则跳过
            if len(line) == 0:
                continue
            # 统计每一字出现的个数
            for characer in line:
                content_len += 1
                # 如果字符第一次出现 加入到字符数组中
                if not characer in characers:
                    characers.append(characer)
                # 如果是字符第一次出现 加入到字典中
                if characer not in rate:
                    rate[characer] = 1
                else:
                    rate[characer] += 1

                record_delta = 1024
                if (content_len / record_delta) > record_num:
                    print(content_len)
                    current_entropy = calculate_entropy(rate, content_len)
                    with open("entropy_ch.txt", "a") as fw:
                        write_str = str((content_len / record_delta) * 3 / 1024) + ":" + str(current_entropy) + "\n"
                        fw.write(write_str)
                        print("recorded file size : ", (content_len / record_delta) * 3 / 1024, " MB")

                    record_num += 1


        # 按出现次数排序
        sort_rate = sorted(rate.items(), key=lambda e: e[1], reverse=True)
        print('语料库共有{0}个字'.format(content_len))
        print('其中有{0}个不同的字'.format(len(characers)))

        k = 0
        for i in sort_rate:
            if i[1] > 5000:
                print("[", i[0], "] 频次为 ", float(i[1]), "次")
                k += 1
            if k == 16:
                break

        current_entropy = calculate_entropy(rate, content_len)
        with open("entropy_ch.txt", "a") as fw:
            write_str = str((content_len / record_delta) * 3) + ":" + str(current_entropy) + "\n"
            fw.write(write_str)

        record_num += 1

        return current_entropy

if __name__ == "__main__":
    # 清洗爬虫下来的数据，计算熵并打印输出
    # 打开文件
    with open('ch_content.txt', 'r', encoding='utf-8') as f:
        txt = f.read()
        txt = re.sub(r'[^\u4e00-\u9fa5]+', "", txt)

    # 把清洗过的文本写回去
    with open('ch_content.txt', 'w', encoding='utf-8') as f:
        f.write(str(txt))
        f.close()

    entropy = record_entropy("ch_content.txt")

    print('Entropy of Chinese is {0}'.format(entropy))
