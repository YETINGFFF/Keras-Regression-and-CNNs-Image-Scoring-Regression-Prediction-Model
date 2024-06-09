# USAGE
# python cnn_regression.py 

# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#from pyimagesearch import datasets
import numpy as np
import argparse
import locale
import os
import shutil 
import cv2
import numpy as np
from pathlib import Path
import itertools

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

#from pyimagesearch import datasets
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.utils import plot_model




EPOCHS_NUM=200
# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset

print("[INFO] loading house attributes...")
inputPath =  "scores.csv"
df = pd.read_csv(inputPath, sep=",")
print(df.head())
image_list=df.iloc[:,0]
score=df.iloc[:,1]
#__________________________________________________________________________________________
print(image_list.shape)
print(score.shape)


trainingImages=[]

for recordIndex in image_list.to_list():
    dirOfImages = os.path.join("test", str(recordIndex) + ".png")
    #print(dirOfImages)
    
    # 尝试加载图像
    img = cv2.imread(dirOfImages)
    
    if img is not None:
        # 图像加载成功，执行 resize 操作
        img = cv2.resize(img, (64, 64))
        trainingImages.append(img)
    else:
        print(f"Error: Failed to load image at path {dirOfImages}")


train_data=np.array(trainingImages)
print(train_data.shape)
print(score.shape)



(trainY, testY, trainX, testX) = train_test_split(score,train_data, test_size=0.25, random_state=42)



maxPrice = trainY.max()
print("maxPrice={}".format(maxPrice))
input("press any key")

trainY=trainY.values
trainY = trainY / maxPrice

testY=testY.values
testY = testY / maxPrice





inputShape = (64, 64, 3)
chanDim = -1
# define the model input
inputs = Input(shape=inputShape)
# CONV => RELU => BN => POOL
x = Conv2D(16, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# CONV => RELU => BN => POOL
x = Conv2D(32, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# CONV => RELU => BN => POOL
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# flatten the volume, then FC => RELU => BN => DROPOUT
x = Flatten()(x)
x = Dense(16)(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Dropout(0.5)(x)

# apply another FC layer, this one to match the number of nodes
# coming out of the MLP
x = Dense(4)(x)
x = Activation("relu")(x)
x = Dense(1, activation="linear")(x)

# construct the CNN
model = Model(inputs, x)




#model = models.create_cnn(64, 64, 3, regress=True)
model.summary()






opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
history=model.fit(trainX, trainY, validation_data=(testX, testY),epochs=EPOCHS_NUM, batch_size=8)

# make predictions on the testing data


#--------------------------------------------------------------
from PIL import Image
import os
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

my_out_data=np.array(resized_images)
my_predict_out=model.predict(my_out_data)



pd.DataFrame(my_predict_out*maxPrice).to_csv("my_predict.csv")







model.save("housePrice.keras2")
print("[INFO] model saved to housePrice.keras2")

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testX)

from sklearn.metrics import mean_absolute_error
print("-----------mape-----------------------")
print(mean_absolute_error(testY,preds))
print("-----------mape-----------------------")
# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(score.mean(), grouping=True),
	locale.currency(score.std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


#readjust house prices
testY=testY*maxPrice
preds=preds*maxPrice


validationLoss=(history.history['val_loss'])
trainingLoss=history.history['loss']




#------------------------------------------------
# Plot training and validation accuracy per epoch
epochs   = range(len(validationLoss)) # Get number of epochs
 #------------------------------------------------
plt.plot  ( epochs,     trainingLoss ,label="Training Loss")
plt.plot  ( epochs, validationLoss, label="Validation Loss" )
plt.title ('Training and validation loss')
plt.xlabel("Epoch #")
plt.ylabel("Loss")
fileToSaveAccuracyCurve="plot_acc.png"
plt.savefig("plot_acc.png")
print("[INFO] Loss curve saved to {}".format("plot_acc.png"))
plt.legend(loc="upper right")
plt.show()





#plot curves (Actual vs Predicted)
plt.plot  ( testY ,label="Actual price")
plt.plot  ( preds, label="Predicted price" )
plt.title ('Security scores')
plt.xlabel("Point #")
plt.ylabel("Price")
plt.legend(loc="upper right")
plt.savefig("scores.png")
plt.show()
print("[INFO] predicted vs actual price saved to scores.png")



