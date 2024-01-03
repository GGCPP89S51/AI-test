import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(root=data_dir, transform=transform)
        self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        class_name = self.class_names[label]
        return img, class_name  # 返回图像和类别名称

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        
        
        # 定义全连接层
        self.fc1 = nn.Linear(32 * 25 * 25, 512)  # 注意更新這裡的輸入維度
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 卷积层1 + 激活函数 + 池化层
        x = self.pool(self.relu(self.conv1(x)))
        # 卷积层2 + 激活函数 + 池化层
        x = self.pool(self.relu(self.conv2(x)))
        # 新增的卷積層3 + 激活函数 + 池化层
        
        
        # 展平
        x = x.view(-1, 32 * 25 * 25)  # 注意更新這裡的輸入維度
        # 全连接层1 + 激活函数
        x = self.relu(self.fc1(x))
        # 全连接层2 (输出层)
        x = self.fc2(x)
        
        return x


# 定义 collate_fn 函数
def collate_fn(batch):
    # 从批次中提取图像和类别名称
    images, class_names = zip(*batch)

    # 将图像调整为相同的大小，并传递 antialias 参数
    resized_images = [Resize((100, 100), antialias=True)(image) for image in images]

    # 将调整后的图像和类别名称返回
    return torch.stack(resized_images), class_names

# 指定数据集路径和转换
custom_dataset = CustomDataset(data_dir='raw-img/', transform=ToTensor())

# 使用 DataLoader 进行批次载入，指定 collate_fn
data_loader = DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# 获取数据集中类别的数量
num_classes = len(custom_dataset.class_names)

# 初始化模型并将模型移到 GPU 上
model = SimpleCNN(num_classes=num_classes).to(device)

# 定义训练的总轮数
num_epochs = 10

# 检查是否存在已经训练好的模型文件
model_save_path = "simple_model.pth"
if os.path.exists(model_save_path):
    # 如果存在模型文件，加载模型参数
    model.load_state_dict(torch.load(model_save_path))
    print("Model loaded from existing file.")
    sample_indices = random.sample(range(len(custom_dataset)), 10)
    sample_images = [custom_dataset[i][0] for i in sample_indices]
    sample_labels = [custom_dataset[i][1] for i in sample_indices]

    # 将图像移到 GPU 上，并进行预测
    sample_images = torch.stack([Resize((100, 100), antialias=True)(image) for image in sample_images]).to(device)
    predictions = model(sample_images)
        # 显示预测结果
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.imshow(sample_images[i].permute(1, 2, 0).cpu().numpy())
        predicted_label = custom_dataset.class_names[predictions[i].argmax().item()]
        plt.title(f"Actual Label: {sample_labels[i]}\nPredicted Label: {predicted_label}")
        plt.axis('off')
        plt.show()
else:
    # 如果不存在模型文件，进行训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 存储训练损失的列表
    for epoch in range(num_epochs):
        train_losses = []  # 清除训练损失列表
        model.train()  # 设置模型为训练模式

        # 迭代数据集的每个批次
        for batch in data_loader:
            # 获取批次数据，并将数据移到 GPU 上
            images, class_names = batch[0].to(device), batch[1]
            labels = torch.tensor([custom_dataset.class_names.index(class_name) for class_name in class_names]).to(device)

            # 在每个批次上进行训练
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 存储训练损失
            train_losses.append(loss.item())

        # 随机选择四张图像并显示预测结果
        sample_indices = random.sample(range(len(custom_dataset)), 10)
        sample_images = [custom_dataset[i][0] for i in sample_indices]
        sample_labels = [custom_dataset[i][1] for i in sample_indices]

        # 将图像移到 GPU 上，并进行预测
        sample_images = torch.stack([Resize((100, 100), antialias=True)(image) for image in sample_images]).to(device)
        predictions = model(sample_images)

        # 在 CPU 上绘制训练损失的图形
        plt.plot(train_losses, label=f'Training Loss (Epoch {epoch+1})')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # 显示预测结果
        for i in range(10):
            plt.subplot(5, 2, i+1)
            plt.imshow(sample_images[i].permute(1, 2, 0).cpu().numpy())
            predicted_label = custom_dataset.class_names[predictions[i].argmax().item()]
            plt.title(f"Actual Label: {sample_labels[i]}\nPredicted Label: {predicted_label}")
            plt.axis('off')

        plt.show()


    # 保存训练好的模型
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

