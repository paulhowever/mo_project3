"""Архитектуры нейросетей для CIFAR-10: ResNet-like (Model1) и plain (Model2)."""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Два conv 3×3 с BN и ReLU; опционально shortcut (ResNet)."""

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        use_shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_shortcut = use_shortcut
        self.shortcut = nn.Sequential()
        if use_shortcut and (stride != 1 or in_planes != planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_shortcut:
            out = out + self.shortcut(x)
        return torch.relu(out)


class CifarResNetLike(nn.Module):
    """Три группы [64,128,256] по 2 BasicBlock; GAP + FC."""

    def __init__(self, num_classes: int = 10, use_shortcut: bool = True) -> None:
        super().__init__()
        self.in_planes = 64
        self.use_shortcut = use_shortcut
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, self.use_shortcut))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def Model1() -> nn.Module:
    """ResNet-like с shortcut."""
    return CifarResNetLike(num_classes=10, use_shortcut=True)


def Model2() -> nn.Module:
    """Та же топология без shortcut (plain)."""
    return CifarResNetLike(num_classes=10, use_shortcut=False)


def build_model(name: str) -> nn.Module:
    """Собрать модель по имени для CLI."""
    key = name.lower().strip()
    if key == "model1":
        return Model1()
    if key == "model2":
        return Model2()
    raise ValueError(f"Неизвестная модель: {name}. Ожидается model1 или model2.")
