import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import models
from tqdm import tqdm
import os
from torchvision.models import mobilenet_v3_large
import clip
import logging
import sys


MODEL_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_clip.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# 自动检测计算设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {DEVICE}")

# 日志
LOG_PATH = os.path.join("output_cnn", "train.log")
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
        self.root = Path(root)
        self.split = split
        self.transform = transform  # 图像预处理参数

        # 读取CSV，根据CSV内容对图片进行类别的划分
        df = pd.read_csv(self.root / "train_test_val_split.csv")
        df["path"] = df["path"].astype(str)

        # 定义类别映射
        self.classes = ["dog", "horse", "elephant", "butterfly",
                        "chicken", "cat", "cow", "sheep", "spider", "squirrel"]
        self.classes_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 根据类别筛选数据
        df = df[df["split"] == self.split].reset_index(drop=True)

        # 可选：为测试程序可行性和正确性，采用少量训练集训练，节省时间
        # if self.split == "train":
        #    max_per_class = 600
        #     df = (
        #         df.groupby("label", group_keys=False)
        #           .apply(lambda x: x.sample(
        #               n=min(len(x), max_per_class),
        #               random_state=42
        #           ))
        #           .reset_index(drop=True)
        # )

        # 存储路径和标签
        self.paths = [self.root / p for p in df["path"].tolist()]
        self.labels = [self.classes_to_idx[c] for c in df["label"].tolist()]

    def __len__(self):
        return len(self.paths)

    # 根据给定索引，读取并返回对应位置的单个样本
    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]

        # 必须转为 RGB，因为我们的部分输入图片是RGBA的png格式，直接读取会有四个通道
        img = Image.open(path).convert("RGB")

        # 图像预处理
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)


# 数据导入函数
def data_load(root):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),  # 水平翻转数据增强
        transforms.RandomRotation(15),  # 随机旋转15度以内
        # 随机调整图像的亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),  # 转换维度(224*224*3→3*224*224)并归一化至 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 验证/测试集预处理：严谨起见，不做随机增强，仅做标准化
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 每次送入的数据样本量
    batch_size = 128

    # shuffle = True将数据的索引随机打乱
    train_dataset = AnimalsDataset(root=root, split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = AnimalsDataset(root=root, split="val", transform=val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = AnimalsDataset(root=root, split="test", transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_weights = None
    # 可选：根据每个类别的图像数量进行对应概率的选择，以避免各类别训练样本数量不同的问题
    # # 所有类别数量初始化为0
    # class_counts = [0] * len(train_dataset.classes)
    # for label in train_dataset.labels:
    #     class_counts[label] += 1
    # class_weights = [1.0 / count if count > 0 else 0.0 for count in class_counts]
    # # 为一张图片根据类别加权
    # sample_weights = [class_weights[label] for label in train_dataset.labels]
    # # 创建加权采样器 replacement = True表示可以重复抽取
    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

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

#



# 准确率评估
def evaluate(model, dataloader, epoch, num_epochs):
    model.eval()    # 切换到评估模式
    correct_count = 0
    total_count = 0

    # 测试环节不进行参数更新
    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            leave=False
        )
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)

            # 每行选出分数最大的索引作为预测结果
            predicted = logits.argmax(dim=1)

            # 统计总数和正确数以计算正确率
            total_count += labels.numel()
            correct_count += (predicted == labels).sum().item()

    return correct_count / total_count


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

    plt.savefig('output/training_curves.png')
    # plt.show()


# 验证过程
def verify_net(model, val_loader, epoch, num_epochs):
    acc = evaluate(model, val_loader, epoch, num_epochs)
    print(f"验证集准确率为{acc}")
    return acc


# 训练过程
def train_net(model, lr, num_epochs, train_loader, val_loader, class_weights, patience=10):
    if class_weights is not None:
        # 转换为Tensor(可用于GPU上进行计算)
        weights_tensor = torch.tensor(class_weights).to(DEVICE)
        # 定义损失函数(引入了权重处理类别不均衡的问题)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    # 定义优化器，告诉优化器要优化哪些参数，lr为学习率，采用L2正则化权重衰减避免过拟合
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # 余弦退火学习率调整(下降过程平滑一些)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    list_train_loss = []    # 每轮训练的平均损失
    list_train_acc = []     # 训练集准确率
    list_val_acc = []       # 验证集准确率

    # 早停参数设置
    best_val_acc = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # 训练过程
        model.train()   # 切换到训练模式，更新梯度和dropout

        total_loss = 0.0    # 总loss
        correct_train = 0   # 正确个数
        total_train = 0     # 总个数

        # 训练进度可视化
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            leave=False
        )

        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # 前向传播
            logits = model(images)
            # 误差计算
            loss = criterion(logits, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()
            # 正确率和loss统计
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            total_loss += loss.item()

            # 训练进度可视化
            progress_bar.set_postfix(loss=loss.item())

        # 学习率输出
        print(f'第{epoch + 1}次循环:')
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\t当前学习率: {current_lr:.6f}")

        # 训练损失统计
        avg_loss = total_loss / len(train_loader)
        list_train_loss.append(avg_loss)
        print(f"\t训练平均loss为{avg_loss}")

        # 训练准确率统计
        train_acc = correct_train / total_train
        list_train_acc.append(train_acc)
        print(f'\t训练正确率为{train_acc}')

        # 验证集准确率统计
        verify_acc = verify_net(model, val_loader, epoch, num_epochs)
        list_val_acc.append(verify_acc)

        # 采用早停机制控制过拟合，如果连续patience次正确率没有提升，则停止
        if verify_acc > best_val_acc:
            best_val_acc = verify_acc
            early_stop_counter = 0      # 早停计数
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc
            }, BEST_MODEL_PATH)

            print(f"\t保存最佳模型（epoch {epoch + 1}, val_acc={best_val_acc:.4f}）")
        else:
            early_stop_counter += 1
            print(f"\t验证集未提升，EarlyStopping计数: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"\n早停触发：连续 {patience} 个 epoch 验证集未提升，停止训练")
                break

        scheduler.step()

    return model, list_train_acc, list_val_acc, list_train_loss


# 绘制混淆矩阵
def plot_confusion_matrix(model, test_loader, class_names):
    # 预测测试集，统计所有预测标签的个数
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(
            test_loader,
            desc="Testing",
            leave=False
        )

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 绘图函数进行混淆矩阵绘制
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("output/confusion_matrix.png")
    plt.show()


# 测试
def test_net(model, test_loader, test_dataset):
    acc = evaluate(model, test_loader, 0, 1)
    print(f"测试集准确率为{acc}")
    plot_confusion_matrix(model, test_loader, class_names=test_dataset.classes)


def main():
    # 数据导入
    print("数据开始导入。。。")
    data_class = data_load("data/Animals-10")
    train_loader = data_class["train_loader"]
    test_dataset = data_class["test_dataset"]
    val_loader = data_class["val_loader"]
    test_loader = data_class["test_loader"]
    class_weights = data_class["class_weights"]
    print("数据导入成功！\n")

    # 模型
    num_classes = 10
    model = CLIPClassifier(num_classes).to(DEVICE)

    if os.path.exists(BEST_MODEL_PATH):
        print(f"检测到已存在最佳模型：{BEST_MODEL_PATH}，直接加载并测试")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.to(DEVICE)

    else:
        print("训练开始。。。")
        model, list_train_acc, list_val_acc, list_train_loss = train_net(
            model,
            lr=0.1,
            num_epochs=10,
            train_loader=train_loader,
            val_loader=val_loader,
            patience=10,
            class_weights=class_weights
        )
        # 绘图
        draw_train_plot(list_train_acc, list_val_acc, list_train_loss)
        print("训练结束!\n")

    # 测试
    print("测试开始。。。")
    test_net(model, test_loader, test_dataset)
    print("测试结束！\n")
    return 0


if __name__ == "__main__":
    main()