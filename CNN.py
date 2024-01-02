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

class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3 * 200 * 300, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 定义 collate_fn 函数
def collate_fn(batch):
    # 从批次中提取图像和类别名称
    images, class_names = zip(*batch)

    # 将图像调整为相同的大小，并传递 antialias 参数
    resized_images = [Resize((300, 200), antialias=True)(image) for image in images]

    # 将调整后的图像和类别名称返回
    return torch.stack(resized_images), class_names

# 指定数据集路径和转换
custom_dataset = CustomDataset(data_dir='raw-img/', transform=ToTensor())

# 使用 DataLoader 进行批次载入，指定 collate_fn
data_loader = DataLoader(dataset=custom_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# 获取数据集中类别的数量
num_classes = len(custom_dataset.class_names)

# 初始化模型并将模型移到 GPU 上
model = SimpleModel(num_classes=num_classes).to(device)

# 定义训练的总轮数
num_epochs = 100

# 检查是否存在已经训练好的模型文件
model_save_path = "simple_model.pth"
if os.path.exists(model_save_path):
    # 如果存在模型文件，加载模型参数
    model.load_state_dict(torch.load(model_save_path))
    print("Model loaded from existing file.")
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
        sample_indices = random.sample(range(len(custom_dataset)), 4)
        sample_images = [custom_dataset[i][0] for i in sample_indices]
        sample_labels = [custom_dataset[i][1] for i in sample_indices]

        # 将图像移到 GPU 上，并进行预测
        sample_images = torch.stack([Resize((300, 200), antialias=True)(image) for image in sample_images]).to(device)
        predictions = model(sample_images)

        # 在 CPU 上绘制训练损失的图形
        plt.plot(train_losses, label=f'Training Loss (Epoch {epoch+1})')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        

        # 显示预测结果
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(sample_images[i].permute(1, 2, 0).cpu().numpy())
            predicted_label = custom_dataset.class_names[predictions[i].argmax().item()]
            plt.title(f"Actual Label: {sample_labels[i]}\nPredicted Label: {predicted_label}")
            plt.axis('off')

        

    # 保存训练好的模型
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

