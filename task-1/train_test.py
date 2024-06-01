
import argparse
import datetime
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
#from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split

#from PIL import Image
import data_transform
import settings
from load import CUB


# 5. 初始化TensorBoard  
# 训练循环  
def train_model(model, train_loader, optimizer, criterion, num_epochs, validation_loader=None, writer=None):  
    best_acc = 0.0
    for epoch in range(num_epochs):  
        model.train()  # 设置模型为训练模式  
        running_loss = 0.0  
          
        for inputs, labels in train_loader:  
            inputs = inputs.to(device)  
            labels = labels.to(device)  
            optimizer.zero_grad()  # 清除之前的梯度  
            outputs = model(inputs)  # 前向传播  
            loss = criterion(outputs, labels)  # 计算损失  
            loss.backward()  # 反向传播  
            optimizer.step()  # 更新权重  
              
            running_loss += loss.item() * inputs.size(0)  # 累积损失  
          
        epoch_loss = running_loss / len(train_loader.dataset)  # 计算平均损失  
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')  
  
        if validation_loader is not None:  
            val_loss, val_acc = validate_model(model, validation_loader, criterion)  
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')  

            
            if writer is not None:  
                writer.add_scalar('Loss/train', epoch_loss, epoch)  
                writer.add_scalar('Loss/val', val_loss, epoch)  
                writer.add_scalar('Accuracy/val', val_acc, epoch)  
            
            if val_acc > best_acc:  
                best_acc = val_acc  
                # 保存最佳模型（例如，使用torch.save）  
                torch.save(model.state_dict(), './model/best_model.pt')  
    # 进行测试
    _, test_acc = validate_model(model,test_dataloader,criterion)
    print(f'Test Accuracy: {test_acc:.4f}')
  
### 验证模型性能  
  


def validate_model(model, validation_loader, criterion):  
    model.eval()  # 设置模型为评估模式  
    val_loss = 0.0  
    correct = 0  
    total = 0  
      
    with torch.no_grad():  # 不需要计算梯度  
        for inputs, labels in validation_loader:  
            inputs = inputs.to(device)  
            labels = labels.to(device)  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            val_loss += loss.item() * inputs.size(0)  
            _, predicted = torch.max(outputs, 1)  
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  
      
    val_loss /= len(validation_loader.dataset)  
    val_acc = 100 * correct / total  
    return val_loss, val_acc
  

def test_model(model, test_loader, criterion):  
    test_loss, test_acc = validate_model(model, test_loader, criterion)  
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')  
    return test_acc

# 2. 数据预处理和加载  

# 定义数据转换  
def get_dataloader(ranseed = 1234):
    train_transforms = data_transform.Compose([
        #transforms.ToPILImage(),
        data_transform.ToCVImage(),
        data_transform.RandomResizedCrop(settings.IMAGE_SIZE),
        data_transform.RandomHorizontalFlip(),
        data_transform.ToTensor(),
        data_transform.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    test_transforms = data_transform.Compose([
        data_transform.ToCVImage(),
        data_transform.Resize(settings.IMAGE_SIZE),
        data_transform.ToTensor(),
        data_transform.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    train_raw_dataset = CUB(
            settings.DATA_PATH,
            train=True,
            transform=train_transforms,
            target_transform=None
        )

    torch.manual_seed(ranseed)  # 设置PyTorch的随机种子 
    np.random.seed(ranseed) 
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(ranseed)  # 如果使用GPU，也设置GPU的随机种子 
    train_size = int(len(train_raw_dataset)*(1-settings.VAL_RATIO))
    val_size = len(train_raw_dataset) - train_size  
    train_dataset, val_dataset = random_split(train_raw_dataset, [train_size, val_size])  

    train_dataloader = DataLoader(
        train_dataset,
        batch_size= settings.BITCH_SIZE,
        num_workers=0,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size= settings.BITCH_SIZE,
        num_workers=0,
        shuffle=False
    )


    test_dataset = CUB(
            settings.DATA_PATH,
            train=False,
            transform=test_transforms,
            target_transform=None
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size= settings.BITCH_SIZE,
        num_workers=0,
        shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader
    
# 3. 加载预训练模型并修改输出层  



# 加载预训练的ResNet-18模型  

def modified_train():
    log_dir = f"logs/run_modified_{num_epochs}_{m_lr_new}_{m_momentum}_{m_weight_decay}_{d_lr}_{d_momentum}_{d_weight_decay},{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir)
    
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  


    # 修改输出层（fc层）  
    # 首先获取当前fc层的输入特征数  
    in_features = resnet.fc.in_features  
    
    # 创建一个新的fc层，输入特征数为之前的fc层的输入特征数，输出特征数为200  
    new_fc = nn.Linear(in_features, 200)  
    
    # 替换模型的fc层  


    resnet.fc = new_fc  


    # 切换到评估模式，确保BN层使用预训练时的统计信息  
    pretrained_model = resnet.to(device)
    pretrained_model.eval()

    # 设置微调参数  
    # 定义一个优化器，为不同的层设置不同的学习率  
    # 我们可以使用参数组（parameter groups）来实现这一点  
    # 假设我们使用SGD优化器  
    
    # 获取新添加的fc层的参数  
    new_fc_parameters = list(filter(lambda p: p.requires_grad, pretrained_model.fc.parameters()))  
    
    # 获取其余预训练层的参数  
    base_parameters = list(filter(lambda p: p.requires_grad and not any(id(p) == id(q) for q in new_fc_parameters), pretrained_model.parameters()))  
    
    # 定义优化器，为不同的参数组设置不同的学习率  
    optimizer = torch.optim.SGD([  
        {'params': base_parameters, 'lr': m_lr_pre},  # 其余预训练层的学习率  
        {'params': new_fc_parameters, 'lr': m_lr_new}  # 新fc层的学习率  
    ], momentum=m_momentum, weight_decay=m_weight_decay)  

    train_model(pretrained_model, train_dataloader, optimizer, criterion, num_epochs, validation_loader=val_dataloader, writer=writer)  
    writer.close()


def direct_train():
    log_dir = f"logs/run_direct_{num_epochs}_{m_lr_new}_{m_momentum}_{m_weight_decay}_{d_lr}_{d_momentum}_{d_weight_decay},{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir)
    resnet = models.resnet18(weights=False)  


    # 修改输出层（fc层）  
    # 首先获取当前fc层的输入特征数  
    in_features = resnet.fc.in_features  
    
    # 创建一个新的fc层，输入特征数为之前的fc层的输入特征数，输出特征数为200  
    new_fc = nn.Linear(in_features, 200)  
    
    # 替换模型的fc层  

    resnet.fc = new_fc  


    # 切换到评估模式，确保BN层使用预训练时的统计信息  
    pretrained_model = resnet.to(device)
    pretrained_model.eval()

    # 设置微调参数  
    # 定义一个优化器，为不同的层设置不同的学习率  
    # 我们可以使用参数组（parameter groups）来实现这一点  
    # 假设我们使用SGD优化器  
    
    # 获取新添加的fc层的参数  
    optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=d_lr, momentum=d_momentum, weight_decay=d_weight_decay) 

    train_model(pretrained_model, train_dataloader, optimizer, criterion, num_epochs, validation_loader=val_dataloader, writer=writer)
    writer.close()

# 加载数据集  
train_dataloader, val_dataloader, test_dataloader = get_dataloader()
# 定义损失函数 
criterion = nn.CrossEntropyLoss()  

# 转移到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  


# 训练轮次
# num_epochs = 20

# 更多轮次
num_epochs = 200
  
# 超参数组合
# 学习率为0.01时，效果最好
# hyperparams = [  
#     {'m_lr_pre': 0.00001, 'm_lr_new': 0.1, 'm_momentum': 0.9, 'm_weight_decay': 0, 'd_lr': 0.1, 'd_momentum': 0.9, 'd_weight_decay': 0},
#     {'m_lr_pre': 0.00001, 'm_lr_new': 0.01, 'm_momentum': 0.9, 'm_weight_decay': 0, 'd_lr': 0.01, 'd_momentum': 0.9, 'd_weight_decay': 0}, 
#     {'m_lr_pre': 0.00001, 'm_lr_new': 0.001, 'm_momentum': 0.9, 'm_weight_decay': 0, 'd_lr': 0.001, 'd_momentum': 0.9, 'd_weight_decay': 0},
    
# ]   
# 权值衰减为0时，效果最好
# hyperparams = [
#     {'m_lr_pre': 0.00001, 'm_lr_new': 0.01, 'm_momentum': 0.9, 'm_weight_decay': 0, 'd_lr': 0.01, 'd_momentum': 0.9, 'd_weight_decay': 1e-4},
#     {'m_lr_pre': 0.00001, 'm_lr_new': 0.01, 'm_momentum': 0.9, 'm_weight_decay': 0, 'd_lr': 0.01, 'd_momentum': 0.9, 'd_weight_decay': 1e-3},
#     {'m_lr_pre': 0.00001, 'm_lr_new': 0.01, 'm_momentum': 0.9, 'm_weight_decay': 0, 'd_lr': 0.01, 'd_momentum': 0.9, 'd_weight_decay': 1e-2},
# ]

hyperparams = [
    {'m_lr_pre': 0.00001, 'm_lr_new': 0.01, 'm_momentum': 0.9, 'm_weight_decay': 0, 'd_lr': 0.01, 'd_momentum': 0.9, 'd_weight_decay': 0},
]
if settings.MOD == 'train':
    for hyperparam in hyperparams:  
        m_lr_pre = hyperparam['m_lr_pre']
        m_lr_new = hyperparam['m_lr_new']
        m_momentum = hyperparam['m_momentum']
        m_weight_decay = hyperparam['m_weight_decay']
        d_lr = hyperparam['d_lr']
        d_momentum = hyperparam['d_momentum']
        d_weight_decay = hyperparam['d_weight_decay']
        # tensorboard
        
        # 训练模型
        modified_train()
        direct_train()


if settings.MOD == 'test':
    # 在测试集上评估模型性能
    model = models.resnet18(weights=False)
    model.load_state_dict(torch.load('./model/best_model.pt'))
    _, test_acc = test_model(model, test_dataloader, criterion)
    print(f'Test Accuracy: {test_acc:.4f}')


