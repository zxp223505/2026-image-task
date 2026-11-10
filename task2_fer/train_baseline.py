
import random
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Subset
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import Counter

#定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#转化工具制造
train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪缩放
      transforms.RandomHorizontalFlip(),                      # 左右翻转
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])



transform = transforms.Compose([
      transforms.Resize((224, 224)),   # 必须是元组(224,224)，不是单 int
      transforms.ToTensor(),           # PIL图 -> Tensor，值变到 0~1，通道转到 C×H×W
      transforms.Normalize(            # 标准化：(x - mean) / std
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      ),
])


path = r"C:\Users\zhaopangpang\Desktop\2026task2\fer_data\fer_data\train"


ds = datasets.ImageFolder(path, transform=transforms.ToTensor())

pc_dx1=ImageFolder(path, transform=train_transform)
pc_dx2=ImageFolder(path, transform=transform)

"""
#先取500个，分5份试试
indices = random.sample(range(len(ds)), 500) 
random.shuffle(indices)                              
"""

torch.manual_seed(42)
idx = torch.randperm(len(pc_dx1)).tolist()    # 随机打乱的索引
n_train = int(0.8 * len(idx))
train_ds = Subset(pc_dx1, idx[:n_train])      # 训练：增强
val_ds   = Subset(pc_dx2, idx[n_train:])


# 损失函数：衡量预测和真实标签的差距
#criterion = nn.CrossEntropyLoss()
from collections import Counter
cnt = Counter(pc_dx1.targets)          # 下标 -> 每类图片数
total = len(pc_dx1)
n_cls = 6
weights = torch.zeros(n_cls)
for i in range(n_cls):
      weights[i] = total / (n_cls * cnt[i])   # 少数类权重更大
weights = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 6)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

num_epochs = 7

for epoch in range(num_epochs):      # 循环训练
        print(epoch+1)
        model.train()
        sum_loss = 0.0                 #记录每轮错误率
        for images, labels in train_loader:          
          images = images.to(device)      
          labels = labels.to(device)        # 数据搬到 GPU（和模型同一个 device）
          optimizer.zero_grad()             # ① 清空梯度
          outputs = model(images)           # ② 前向传播：喂图，得预测
          loss = criterion(outputs, labels) # ③ 算损失：预测和真答案差多少
          loss.backward()                   # ④ 反向传播：自动算每个参数的梯度
          optimizer.step()                  # ⑤ 用梯度更新参数
          sum_loss=sum_loss+loss.item()
        print(f"epoch {epoch+1}  avg_loss = {sum_loss/len(train_loader):.4f}")


        model.eval()                 # 切模式
        correct = 0
        total = 0

        with torch.no_grad():
         for images, labels in val_loader:
           images = images.to(device)
           labels = labels.to(device)
           outputs = model(images)              # 前向
           preds = outputs.argmax(dim=1)        # 取预测答案
           correct += (preds == labels).sum().item()
           total += labels.size(0)              
         acc = correct / total
         print(f"epoch {epoch+1}  val_acc = {acc:.4f}")

torch.save(model.state_dict(), "model_8_2.pt")
print("done")




#正式任务
from torch.utils.data import Dataset
from PIL import Image
import os, csv

class TestDataset(Dataset):
      def __init__(self, folder, transform=None):
          self.files = sorted([f for f in os.listdir(folder)
                               if f.lower().endswith(('.png', '.jpg',
  '.jpeg'))])
          self.folder = folder
          self.transform = transform

      def __len__(self):
          return len(self.files)

      def __getitem__(self, idx):
          img = Image.open(os.path.join(self.folder, self.files[idx])).convert('RGB')
          if self.transform:
              img = self.transform(img)
          return img, self.files[idx]      # 图片 + 文件名


test_folder = r"C:\Users\zhaopangpang\Desktop\2026task2\fer_data\fer_data\test"
test_ds = TestDataset(test_folder, transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
print("测试集张数:", len(test_ds)) 

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 6)
model.load_state_dict(torch.load("model_8_2.pt"))
model = model.to(device)
model.eval()

preds = []
with torch.no_grad():
      for images, fnames in test_loader:
          images = images.to(device)
          out = model(images)
          preds.append(out.argmax(dim=1).cpu())
preds = torch.cat(preds, dim=0)

idx2label = {0:0, 1:1, 2:2, 3:5, 4:3, 5:4}
labels = [idx2label[p.item()] for p in preds]

name_list = test_ds.files
name_to_label = {name_list[i]: labels[i] for i in range(len(name_list))}

sample_path = r"C:\Users\zhaopangpang\Desktop\2026task2\submission.csv"
with open(sample_path, 'r') as f:
      reader = csv.reader(f)
      next(reader)                         # 跳过表头
      ids = [row[0] for row in reader]

out_path = r"C:\Users\zhaopangpang\Desktop\2026task2\my_submission2.csv"
with open(out_path, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['ID', 'Emotion'])
      for fid in ids:
          writer.writerow([fid, name_to_label[fid]])
print(f"写出 {len(ids)} 行到 {out_path}")