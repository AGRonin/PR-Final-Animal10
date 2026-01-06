from torch import nn
from torchvision.models import mobilenet_v3_large


# 使用mobilenetV3Large深度神经网络（冻结）
class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV3LargeClassifier, self).__init__()
        # 加载预训练的MobileNetV3 Large模型
        self.base_model = mobilenet_v3_large(pretrained=pretrained)
        # 设计最后一层分类器，使其符合最后我们任务（最后分为10类）
        self.base_model.classifier=nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),     # 非线性函数
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

        # 冻结主干网络所有参数（将requires_grad设为False）
        for param in self.base_model.parameters():
            param.requires_grad = False
        # 第二步：解冻分类器参数，仅训练分类器
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)


# 使用mobilenetV3Large深度神经网络（非冻结）
class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MobileNetV3LargeClassifier, self).__init__()
        # 加载MobileNetV3 Large模型
        self.base_model = mobilenet_v3_large(pretrained=pretrained)
        # 设计最后一层分类器，使其符合最后我们任务（最后分为10类）
        self.base_model.classifier=nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),     # 非线性函数
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        return self.base_model(x)