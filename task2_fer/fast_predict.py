import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os, csv

# ===== 配置：默认单折最快(约4分钟)；想冲高把 FOLDS 改成 [1,2,3,4,5](约20分钟) =====
BACKBONE = 'resnet50'
BATCH = 128
IMG = 224
FOLDS = [5]   # 只用 fold5，最快。要更稳可改 [3,5] 或 [1,2,3,4,5]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TestDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([f for f in os.listdir(folder)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.folder = folder
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.folder, self.files[i])).convert('RGB')
        return transform(img), self.files[i]

def make_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(2048, 6)
    return m

WT_DIR = r"C:\Users\zhaopangpang\Desktop\2026task2\modle\最终3\最终3"
TEST_DIR = r"C:\Users\zhaopangpang\Desktop\2026task2\fer_data\fer_data\test"
SAMPLE = r"C:\Users\zhaopangpang\Desktop\2026task2\submission.csv"
OUT = r"C:\Users\zhaopangpang\Desktop\2026task2\my_submission_resnet50_fast.csv"

test_ds = TestDataset(TEST_DIR)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
print("test images:", len(test_ds))

sum_logits = None
for fi in FOLDS:
    m = make_model()
    m.load_state_dict(torch.load(os.path.join(WT_DIR, f"fold{fi}_resnet50.pt")))
    m = m.to(device)
    m.eval()
    logits = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            logits.append(m(imgs.to(device)).cpu())
    logits = torch.cat(logits, dim=0)
    sum_logits = logits if sum_logits is None else sum_logits + logits
    print(f"fold {fi} done")

preds = sum_logits.argmax(dim=1)
idx2label = {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4}
labels = [idx2label[p.item()] for p in preds]
name2label = {test_ds.files[i]: labels[i] for i in range(len(test_ds.files))}

with open(SAMPLE, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    ids = [row[0] for row in reader]

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'Emotion'])
    for fid in ids:
        w.writerow([fid, name2label[fid]])
print("wrote", len(ids), "rows to", OUT)
