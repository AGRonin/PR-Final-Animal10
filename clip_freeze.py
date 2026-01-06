import torch
import clip
import os
from torch import nn
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


# 数据导入类设计
class AnimalsDataset(Dataset):
    def __init__(self, root: str, split: str, transform: transforms.Compose = None):
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
        df = df[df["split"] == self.split].reset_index(drop=True)
        #if self.split == "train":
        #    max_per_class = 600   
        #     df = (
        #         df.groupby("label", group_keys=False)
        #           .apply(lambda x: x.sample(
        #               n=min(len(x), max_per_class),
        #               random_state=42
        #           ))
        #           .reset_index(drop=True)
        # )
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
        transforms.RandomHorizontalFlip(),  # 水平翻转数据增强
        transforms.ToTensor(),  # 转换通道
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试集预处理
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
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


class CLIPClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CLIPClassifier, self).__init__()
        #加载预训练参数结果
        self.clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

        #冻结全部参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        #解冻clip最后一层参数
        for param in self.clip_model.visual.transformer.resblocks[-1].parameters():
            param.requires_grad = True

        #将最后一层从512转换为类别数
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        #将图片变成512维的特征向量(为什么是512维 和clip模型有关，clip的核心思想是把图像和文本映射到同一个向量空间里)
        image_features = self.clip_model.encode_image(x)
        #进行分类
        logits = self.classifier(image_features)
        return logits


# 准确率评估
def evaluate(model, dataloader):
    model.eval()
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
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

    plt.savefig('output_clip_freeze/training_curves.png')
    plt.show()


# 验证过程
def verify_net(model, val_loader):
    acc = evaluate(model, val_loader)
    print(f"验证集准确率为{acc}")
    return acc


# 训练过程
def train_net(model, lr, num_epochs, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    list_train_loss = []
    list_train_acc = []
    list_val_acc = []
    last_val_acc = 0
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # 训练过程
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # 前向传播
            logits = model(images)
            # 误差计算
            loss = criterion(logits, labels)
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

        # 验证准确率统计
        verify_acc = verify_net(model, val_loader)
        #if last_val_acc > verify_acc:
        #    continue
        #else:
        last_val_acc = verify_acc
        list_val_acc.append(verify_acc)
        if verify_acc > best_val_acc:
            best_val_acc = verify_acc
            torch.save(model.state_dict(), 'best_model_clip_freeze.pth')

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
    plt.savefig("output_clip_freeze/confusion_matrix.png")
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
    print("数据导入成功！\n")

    # 模型
    num_classes = 10
    model = CLIPClassifier(num_classes).to(DEVICE)
    best_model_path = "best_model_clip_freeze.pth"
    # 训练+验证
    if os.path.exists(best_model_path):
        print(f"检测到已有最佳模型: {best_model_path}，直接加载跳过训练")
        model.load_state_dict(torch.load(best_model_path))
        model.to(DEVICE)
        list_train_acc, list_val_acc, list_train_loss = [], [], []
    else:
        print("训练开始。。。")
        model, list_train_acc, list_val_acc, list_train_loss = train_net(model, 1e-4,10 , train_loader, val_loader)
        model.load_state_dict(torch.load(best_model_path))
        model.to(DEVICE)
        print("训练结束!\n")
    # 绘图
    draw_train_plot(list_train_acc, list_val_acc, list_train_loss)

    # 测试
    print("测试开始。。。")
    test_net(model, test_loader, test_dataset)
    print("测试结束！\n")
    return 0


if __name__ == "__main__":
    os.makedirs("output_clip_freeze", exist_ok=True)
    main()
