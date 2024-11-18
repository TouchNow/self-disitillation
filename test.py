import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


def calculate_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算两个张量之间的距离

    Args:
        x: 输入张量1
        y: 输入张量2

    Returns:
        两个张量之间的欧氏距离
    """
    return torch.dist(x, y)


def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """将两个张量相加

    Args:
        a: 输入张量1
        b: 输入张量2

    Returns:
        两个张量的和
    """
    return a + b


def main():
    # 测试MSE损失
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    mse_loss1 = nn.MSELoss()(a, b)
    mse_loss2 = F.mse_loss(a, b)

    # 测试距离计算
    x = torch.randn(4)
    y = torch.randn(4)
    z = torch.randn((2, 4))
    print(f"x和y之间的距离: {calculate_distance(x, y):.4f}")
    print(f"x和z之间的距离: {calculate_distance(x, z):.4f}")


if __name__ == "__main__":
    main()
