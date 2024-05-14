# 打印数字的 Unicode 编码范围
print("数字的 Unicode 编码范围:")
for digit in range(0xFF10, 0xFF1A):
    print(chr(digit), hex(digit))

# 打印全角大写字母的 Unicode 编码范围
print("全角大写字母的 Unicode 编码范围:")
for letter in range(0xFF21, 0xFF3B):
    print(chr(letter), end="")

# 打印全角小写字母的 Unicode 编码范围
print("全角小写字母的 Unicode 编码范围:")
for letter in range(0xFF41, 0xFF5B):
    print(chr(letter))
