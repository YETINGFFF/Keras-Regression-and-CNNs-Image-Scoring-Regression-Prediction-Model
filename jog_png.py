import os
import shutil

# 指定要遍历的文件夹路径
folder_path = "test"

# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.png')):
            # 构建原始文件的完整路径
            original_file_path = os.path.join(root, file)
            
            # 构建新的文件名，将后缀改为.png
            new_file_name = os.path.splitext(file)[0] + ".png"
            
            # 构建新的文件的完整路径
            new_file_path = os.path.join(root, new_file_name)
            
            # 重命名文件并将后缀改为.png
            os.rename(original_file_path, new_file_path)
            print(f"Renamed: {original_file_path} -> {new_file_path}")
