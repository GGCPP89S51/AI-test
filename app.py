import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def imshow(img):
    img = img / 2 + 0.5     # 反標準化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 定義卷積層和池化層
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        
        # 定義全連接層
        self.fc1 = nn.Linear(32 * 25 * 25, 512)  # 注意更新這裡的輸入維度
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 卷積層1 + 激活函數 + 池化層
        x = self.pool(self.relu(self.conv1(x)))
        # 卷積層2 + 激活函數 + 池化層
        x = self.pool(self.relu(self.conv2(x)))
        # 新增的卷積層3 + 激活函數 + 池化層
        
        # 展平
        x = x.view(-1, 32 * 25 * 25)  # 注意更新這裡的輸入維度
        # 全連接層1 + 激活函數
        x = self.relu(self.fc1(x))
        # 全連接層2 (輸出層)
        x = self.fc2(x)
        
        return x

def do_prediction(fileIOObj):
    pil_image = Image.open(fileIOObj).convert('RGB')
    # 將圖像轉換為張量並調整大小以適應我們訓練的模型大小  
    img_loader = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    ts_image = img_loader(pil_image).float()
    ts_image.unsqueeze_(0)
    outputs = net(ts_image)
    _, predicted = torch.max(outputs.data, 1)
    # 預測是一個列表，因此我們必須獲取第一個元素    
    return classes[predicted[0]]

PATH = './simple_model.pth'

classes = ('boosy', 'Butterfly', 'cat', 'chicken', 'dog', 'elephants', 'horse', 'sheep', 'spider', 'squirrel')

uploaded_file = st.file_uploader("上傳圖像", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='上傳的圖像', use_column_width=True)
    st.write("")
    st.write("正在分類...")

    net = SimpleCNN()
    net.load_state_dict(torch.load(PATH))
    
    for parameter in net.parameters():
        parameter.requires_grad = False

    prediction = do_prediction(uploaded_file)
    st.write(prediction)
