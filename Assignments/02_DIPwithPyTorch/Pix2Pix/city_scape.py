# 要追加的文件路径
file_path = './datasets/cityscapes/val/'

# 打开文件并追加内容
with open('val_list.txt', 'a', encoding='utf-8') as file:
    for i in range(1, 501):
        file.write(f"{file_path}{i}.jpg\n")

print("内容已成功追加到文件中。")
