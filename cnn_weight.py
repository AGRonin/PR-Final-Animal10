import os
import copy
import sys
import logging
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 自动检测计算设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {DEVICE}")

#日志
LOG_PATH = os.path.join("output_cnn_weight", "train.log")
logger = logging.getLogger("train_logger")
logger.setLevel(logging.INFO)

# 文件日志
file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# 重定向 print → logger（tqdm 不受影响）
class PrintLogger:
    def write(self, message):
        message = message.strip()
        if message:
            logger.info(message)

    def flush(self):
        pass


sys.stdout = PrintLogger()
# 数据导入类设计
class AnimalsDataset(Dataset):
    def __init__(self, root: str, split: str, transform: transforms.Compose = None):
        """
        root: 数据集根目录
        split: 'train', 'val', 或 'test'
        transform: 图像预处理流水线
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        # 读取 CSV 分割文件
        df = pd.read_csv(self.root / "train_test_val_split.csv")
        df["path"] = df["path"].astype(str)

        # 定义类别映射
        self.classes = ["dog", "horse", "elephant", "butterfly",
                        "chicken", "cat", "cow", "sheep", "spider", "squirrel"]
        self.classes_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 筛选对应的数据集 (train/val/test)
        df = df[df["split"] == self.split].reset_index(drop=True)
        if self.split == "train":
            max_per_class = 600   

            df = (
                df.groupby("label", group_keys=False)
                  .apply(lambda x: x.sample(
                      n=min(len(x), max_per_class),
                      random_state=42
                  ))
                  .reset_index(drop=True)
        )
        self.paths = [self.root / p for p in df["path"].tolist()]
        self.labels = [self.classes_to_idx[c] for c in df["label"].tolist()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]
        # 必须转为 RGB，因为我们的部分输入图片是RGBA的png格式，直接读取会有四个通道
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)


# 数据导入函数
def data_load(root):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(),  # 水平翻转数据增强
        transforms.RandomRotation(15),   # 随机旋转15度以内
        #随机调整图像的亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 颜色抖动
        transforms.ToTensor(),  # 转为 Tensor 并归一化至 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试集预处理：严谨起见，不做随机增强，仅做标准化
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #每次送入的数据样本量
    batch_size=128

    train_dataset = AnimalsDataset(root=root, split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = AnimalsDataset(root=root, split="val", transform=val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = AnimalsDataset(root=root, split="test", transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_counts = [0]*len(train_dataset.classes)
    for label in train_dataset.labels:
        class_counts[label] += 1
    class_weights = [1.0/count if count > 0 else 0.0 for count in class_counts]
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    data_class = {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "val_dataset": val_dataset,
        "val_loader": val_loader,
        "test_dataset": test_dataset,
        "test_loader": test_loader,
        "class_weights": class_weights
    }
    return data_class


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1. 特征提取网络 (Backbone)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),nn.MaxPool2d(2),
            # Global Average Pooling: 无论输入尺寸如何，输出特征图均为 1x1
            nn.AdaptiveAvgPool2d((1, 1)), 
        )
        
        # 2. 分类器 (Head)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.2), # dropout 随机让部分神经元失活，防止过拟合
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 准确率评估
def evaluate(model, dataloader):
    model.eval()
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            #每行选出分数最大的索引
            predicted = logits.argmax(dim=1)
            total_count += labels.numel()
            correct_count += (predicted == labels).sum().item()
    return correct_count / total_count

def ema_update(current_model, last_model, val_acc_current, val_acc_last, eps=1e-6):
    """
    current_model: 当前训练模型
    last_model: 上一轮模型
    val_acc_current: 当前轮验证集准确率
    val_acc_last: 上一轮验证集准确率
    """
    total = val_acc_current + val_acc_last + eps  # 防止除零
    alpha = val_acc_current / total  # 当前轮权重
    beta = val_acc_last / total      # 上一轮权重

    for p_curr, p_last in zip(current_model.parameters(), last_model.parameters()):
        p_curr.data.copy_(alpha * p_curr.data + beta * p_last.data)

def draw_train_plot(list_train_acc, list_val_acc, list_train_loss):
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(list_train_loss) + 1), list_train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(list_train_acc) + 1), list_train_acc, label='Train Accuracy')
    plt.plot(range(1, len(list_val_acc) + 1), list_val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('output_cnn_w/training_curves.png')
    plt.show()


# 验证过程
def verify_net(model, val_loader):
    acc = evaluate(model, val_loader)
    print(f"验证集准确率为{acc}")
    return acc


# 训练过程
def train_net(model, ema_model, lr, num_epochs, train_loader, val_loader,class_weights):
    weights_tensor = torch.tensor(class_weights).to(DEVICE)
    #定义损失函数
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    #定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    #每轮训练的平均损失
    list_train_loss = []
    #训练集准确率
    list_train_acc = []
    #验证集准确率
    list_val_acc = []
    last_val_acc = 0
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练过程
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # 前向传播
            logits = model(images)
            # 误差计算
            loss = criterion(logits, labels)
            #先将梯度清零，再反向传播，最后更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #累加损失
            total_loss += loss.item()
        print(f'第{epoch + 1}次循环:')

        # 训练损失统计
        avg_loss = total_loss / len(train_loader)
        list_train_loss.append(avg_loss)
        print(f"\t训练平均loss为{avg_loss}")

        # 训练准确率统计
        train_acc = evaluate(model, train_loader)
        list_train_acc.append(train_acc)
        print(f'\t训练正确率为{train_acc}')

        # 验证准确率统计+采用最简单的早停机制控制过拟合
        verify_acc = verify_net(model, val_loader)
        if epoch >=40 and verify_acc < last_val_acc:
            ema_update(model, ema_model, verify_acc, last_val_acc)
            model.load_state_dict(ema_model.state_dict())
            print("\t采用EMA模型进行验证")
            verify_acc = verify_net(model, val_loader)
        if last_val_acc > verify_acc:
            continue
        else:
            last_val_acc = verify_acc
            list_val_acc.append(verify_acc)
        if verify_acc > best_val_acc:
            best_val_acc = verify_acc
            torch.save(model.state_dict(), 'best_model_cnn_w.pth')
        ema_model.load_state_dict(model.state_dict())
        scheduler.step()
    return model, list_train_acc, list_val_acc, list_train_loss


# 绘制混淆矩阵
def plot_confusion_matrix(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("output_cnn_w/confusion_matrix.png")
    plt.show()


# 测试
def test_net(model, test_loader, test_dataset):
    acc = evaluate(model, test_loader)
    print(f"测试集准确率为{acc}")
    plot_confusion_matrix(model, test_loader, class_names=test_dataset.classes)


def main():
    # 数据导入
    print("数据开始导入。。。")
    data_class = data_load("Animals-10")
    train_loader = data_class["train_loader"]
    test_dataset = data_class["test_dataset"]
    val_loader = data_class["val_loader"]
    test_loader = data_class["test_loader"]
    class_weights = data_class["class_weights"]
    print("数据导入成功！\n")

    # 模型
    num_classes = 10
    model = CNN(num_classes).to(DEVICE)
    ema_model = copy.deepcopy(model)
    ema_model.to(DEVICE)
    best_model_path = "best_model_cnn_w.pth"
    if os.path.exists(best_model_path):
        print(f"检测到已有最佳模型: {best_model_path}，直接加载跳过训练")
        model.load_state_dict(torch.load(best_model_path))
        model.to(DEVICE)
        list_train_acc, list_val_acc, list_train_loss = [], [], []
    else:
        print("训练开始。。。")
        model, list_train_acc, list_val_acc, list_train_loss = train_net(model, ema_model,0.01, 120, train_loader, val_loader,class_weights = class_weights)
        print("训练结束!\n")
    # 训练+验证
    print("训练开始。。。")
    # train_net(model, lr, num_epochs, train_loader, val_loader)
    # return model, list_train_acc, list_val_acc
    print("训练结束!\n")

    # 绘图
    # draw_train_plot(list_train_acc, list_val_acc, list_train_loss)
    draw_train_plot(list_train_acc, list_val_acc, list_train_loss)

    # 测试
    print("测试开始。。。")
    # test_net(model, test_loader, test_dataset)
    test_net(model, test_loader, test_dataset)
    print("测试结束！\n")
    return 0


if __name__ == "__main__":
    os.makedirs("output_cnn_w", exist_ok=True)
    main()
