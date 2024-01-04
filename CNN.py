import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os


# 檢查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義數據集類
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(root=data_dir, transform=transform)
        self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        class_name = self.class_names[label]
        return img, class_name  # 返回圖像和類別名稱



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


# 定義 collate_fn 函數
def collate_fn(batch):
    # 從批次中提取圖像和類別名稱
    images, class_names = zip(*batch)

    # 將圖像調整為相同的大小，並傳遞 antialias 參數
    resized_images = [Resize((100, 100), antialias=True)(image) for image in images]

    # 將調整後的圖像和類別名稱返回
    return torch.stack(resized_images), class_names

# 指定資料集路徑和轉換
custom_dataset = CustomDataset(data_dir='raw-img/', transform=ToTensor())

# 使用 DataLoader 進行批次載入，指定 collate_fn
data_loader = DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# 獲取資料集中類別的數量
num_classes = len(custom_dataset.class_names)

# 初始化模型並將模型移到 GPU 上
model = SimpleCNN(num_classes=num_classes).to(device)

# 定義訓練的總輪數
num_epochs = 10

# 檢查是否存在已經訓練好的模型檔案
model_save_path = "simple_model.pth"
if os.path.exists(model_save_path):
    # 如果存在模型檔案，加載模型參數
    model.load_state_dict(torch.load(model_save_path))
    print("Model loaded from existing file.")
    sample_indices = random.sample(range(len(custom_dataset)), 10)
    sample_images = [custom_dataset[i][0] for i in sample_indices]
    sample_labels = [custom_dataset[i][1] for i in sample_indices]

    # 將圖像移到 GPU 上，並進行預測
    sample_images = torch.stack([Resize((100, 100), antialias=True)(image) for image in sample_images]).to(device)
    predictions = model(sample_images)
        # 顯示預測結果
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.imshow(sample_images[i].permute(1, 2, 0).cpu().numpy())
        predicted_label = custom_dataset.class_names[predictions[i].argmax().item()]
        plt.title(f"Actual Label: {sample_labels[i]}\nPredicted Label: {predicted_label}")
        plt.axis('off')
        plt.show()
else:
    # 如果不存在模型檔案，進行訓練
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 儲存訓練損失的列表
    for epoch in range(num_epochs):
        train_losses = []  # 清除訓練損失列表
        model.train()  # 設置模型為訓練模式

        # 迭代資料集的每個批次
        for batch in data_loader:
            # 獲取批次資料，並將資料移到 GPU 上
            images, class_names = batch[0].to(device), batch[1]
            labels = torch.tensor([custom_dataset.class_names.index(class_name) for class_name in class_names]).to(device)

            # 在每個批次上進行訓練
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 儲存訓練損失
            train_losses.append(loss.item())

        # 隨機選擇四張圖像並顯示預測結果
        sample_indices = random.sample(range(len(custom_dataset)), 10)
        sample_images = [custom_dataset[i][0] for i in sample_indices]
        sample_labels = [custom_dataset[i][1] for i in sample_indices]

        # 將圖像移到 GPU 上，並進行預測
        sample_images = torch.stack([Resize((100, 100), antialias=True)(image) for image in sample_images]).to(device)
        predictions = model(sample_images)

        # 在 CPU 上繪製訓練損失的圖形
        plt.plot(train_losses, label=f'Training Loss (Epoch {epoch+1})')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # 顯示預測結果
        for i in range(10):
            plt.subplot(5, 2, i+1)
            plt.imshow(sample_images[i].permute(1, 2, 0).cpu().numpy())
            predicted_label = custom_dataset.class_names[predictions[i].argmax().item()]
            plt.title(f"Actual Label: {sample_labels[i]}\nPredicted Label: {predicted_label}")
            plt.axis('off')

        plt.show()


    # 儲存訓練好的模型
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")