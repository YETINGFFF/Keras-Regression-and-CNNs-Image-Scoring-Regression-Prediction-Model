#--------------------------------------------------------------
from PIL import Image
import os
import cv2
# 获取当前文件夹路径
current_directory ="test_photo"

# 用于存储调整大小后的图像的列表
resized_images = []
fILe_path=[]
# 遍历当前文件夹中的所有文件
for filename in os.listdir(current_directory):
    # 检查文件是否是图像文件（这里假设只处理常见的图像格式）
    if filename.endswith(".png"):
        mypath=os.path.join("test_photo", str(filename))
        # 打开图像文件
        img = cv2.imread(mypath)
    
        if img is not None:
        # 图像加载成功，执行 resize 操作
            img = cv2.resize(img, (64, 64))
            resized_images.append(img)
            fILe_path.append(str(filename))

print(fILe_path)