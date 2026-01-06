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
from torchvision.models import resnet18
import os
import sys
import logging

MODEL_DIR = "checkpoints_resnet_pretrain_freeze"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_resnet18.pth")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("output_resnet_pretrain_freeze", exist_ok=True)

# 自动检测计算设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {DEVICE}")

# 日志系统
LOG_PATH = os.path.join("output_resnet_pretrain_freeze", "train.log")
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

# 重定向 print → logger
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

        # 筛选对应的数据集
        df = df[df["split"] == self.split].reset_index(drop=True)
        # 子采样
        if self.split == "train":
            max_per_class = 200

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
        # 必须转为RGB，因为我们的部分输入图片是RGBA的png格式，直接读取会有四个通道
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)


# 数据导入函数
def data_load(root):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), #统一尺寸
        transforms.RandomHorizontalFlip(),  # 以50%概率对图像做水平翻转，防止模型记住方向特征
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机扰动图像的颜色属性，亮度对比度饱和度色度
        transforms.ToTensor(), # 转为Tensor并归一化至[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 对每个通道做标准化（Z-score）
    ])

    # 验证、测试集预处理：仅做标准化
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size=32

    train_dataset = AnimalsDataset(root=root, split="train", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = AnimalsDataset(root=root, split="val", transform=val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = AnimalsDataset(root=root, split="test", transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_class = {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "val_dataset": val_dataset,
        "val_loader": val_loader,
        "test_dataset": test_dataset,
        "test_loader": test_loader
    }
    return data_class

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__() # 调用父类nn.Module的构造函数
        self.backbone = resnet18(weights=True) # 创建一个ResNet18网络

        # 替换分类头
        in_features = self.backbone.fc.in_features # 读取ResNet18原始全连接层的输入特征维度
        self.backbone.fc = nn.Linear(in_features, num_classes) # 用新的全连接层，替换原来ImageNet的1000类分类头

        for name, param in self.backbone.named_parameters():
            # 只让fc层的参数保持可训练，其他参数全部冻结
            if not name.startswith('fc.'):  # 匹配fc.weight和fc.bias
                param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

# 准确率评估
def evaluate(model, dataloader,epoch,num_epochs):
    model.eval()
    correct_count = 0
    total_count = 0
    with torch.no_grad(): # 关闭梯度计算
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            leave=False
        )
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images) # 前向传播得到未经过softmax的原始分类分数
            predicted = logits.argmax(dim=1)
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

    plt.savefig('output_resnet_pretrain_freeze/training_curves.png')
    plt.show()


# 验证过程
def verify_net(model, val_loader,epoch,num_epochs):
    acc = evaluate(model, val_loader,epoch,num_epochs)
    print(f"验证集准确率为{acc}")
    return acc


# 训练过程
def train_net(model, lr, num_epochs, train_loader, val_loader,patience=10):
    criterion = nn.CrossEntropyLoss() # 多分类交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4) # 定义优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=101, gamma=0.1) # 学习率调度器
    list_train_loss = []
    list_train_acc = []
    list_val_acc = []
    best_val_acc = 0
    early_stop_counter = 0
    for epoch in range(num_epochs):
        # 训练过程
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            leave=False
        )
        correct_train = 0
        total_train = 0
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images) # 前向传播
            loss = criterion(logits, labels) # 误差计算
            optimizer.zero_grad() #梯度清零，否则梯度累加
            loss.backward() # 反向传播
            optimizer.step() # 参数更新

            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())
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

        # 验证准确率统计+采用最简单的早停机制控制过拟合
        verify_acc = verify_net(model, val_loader,epoch,num_epochs)
        list_val_acc.append(verify_acc)

        if verify_acc > best_val_acc:
            best_val_acc = verify_acc
            early_stop_counter=0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc
            }, BEST_MODEL_PATH)

            print(f"\t保存最佳模型（epoch {epoch + 1}, val_acc={best_val_acc:.4f}）")
        else:
            early_stop_counter+=1
            print(f"\t验证集未提升，EarlyStopping计数: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"\n早停触发：连续 {patience} 个 epoch 验证集未提升，停止训练")
                break

        scheduler.step()

    return model, list_train_acc, list_val_acc, list_train_loss


# 绘制混淆矩阵
def plot_confusion_matrix(model, test_loader, class_names):
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
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("output_resnet_pretrain_freeze/confusion_matrix.png")
    plt.show()


# 测试
def test_net(model, test_loader, test_dataset):
    acc = evaluate(model, test_loader,0,1)
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
    print("数据导入成功！\n")

    # 模型
    num_classes = 10
    model = ResNetClassifier(num_classes).to(DEVICE)

    if os.path.exists(BEST_MODEL_PATH):
        print("检测到已存在最佳模型，直接加载并测试")

        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"加载模型来自 epoch {checkpoint['epoch']}，val_acc={checkpoint['val_acc']:.4f}")

    else:
        print("训练开始。。。")
        model, list_train_acc, list_val_acc, list_train_loss = train_net(
            model,
            lr=0.1,
            num_epochs=50,
            train_loader=train_loader,
            val_loader=val_loader,
            patience=10
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
