"""25 类交通标志训练入口。"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent
CLASSES_PATH = BASE_DIR / 'classes.txt'
DEFAULT_TRAIN_DIR = Path(r'/home/jovyan/work/datasets/6a55d503fdd9153745ce922b-momodel/train')
DEFAULT_OUTPUT = '/home/jovyan/work/results/model_sample.pth'
NUM_CLASSES = 25
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def normalize_label(value):
    # 兼容磁盘目录中历史遗留的连续空格。
    return ' '.join(str(value).strip().split())


def load_classes():
    indexed = {}
    with CLASSES_PATH.open('r', encoding='utf-8-sig') as file:
        for number, raw in enumerate(file, 1):
            line = raw.rstrip('\r\n')
            if not line:
                continue
            try:
                index_text, name = line.split('\t', 1)
                index = int(index_text)
            except ValueError:
                raise ValueError('classes.txt 第 {0} 行格式错误。'.format(number))
            name = normalize_label(name)
            if index in indexed or not name:
                raise ValueError('classes.txt 第 {0} 行编号重复或类别为空。'.format(number))
            indexed[index] = name
    if sorted(indexed) != list(range(NUM_CLASSES)):
        raise ValueError('classes.txt 必须包含连续编号 0 到 24。')
    classes = [indexed[index] for index in range(NUM_CLASSES)]
    if len(set(classes)) != NUM_CLASSES:
        raise ValueError('classes.txt 存在重复类别名称。')
    return classes


class TrafficSignCNN(nn.Module):
    """与 main.py 完全相同，生成的 checkpoint 可直接用于评分。"""
    def __init__(self, num_classes=NUM_CLASSES):
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class RemappedSubset(Dataset):
    """将 ImageFolder 的字母序标签转换为 classes.txt 的固定编号。"""
    def __init__(self, dataset, indices, mapping):
        self.dataset = dataset
        self.indices = indices
        self.mapping = mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, old_label = self.dataset[self.indices[index]]
        return image, self.mapping[old_label]


def build_loaders(train_dir, classes, image_size, val_ratio, batch_size, seed, workers):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    full_set = datasets.ImageFolder(str(train_dir), transform=transform)
    actual = set(normalize_label(name) for name in full_set.classes)
    expected = set(classes)
    if actual != expected:
        raise ValueError('训练类别目录与 classes.txt 不一致；仅目录中有：{0}；仅 classes.txt 中有：{1}'.format(
            sorted(actual - expected), sorted(expected - actual)))
    mapping = {}
    for name, old_index in full_set.class_to_idx.items():
        mapping[old_index] = classes.index(normalize_label(name))
    counts = [0] * NUM_CLASSES
    for old_label in full_set.targets:
        counts[mapping[old_label]] += 1
    for index, count in enumerate(counts):
        if count < 2:
            raise ValueError('类别 {0} 图片少于 2 张，无法划分验证集。'.format(index))
        print('class_index={0:02d}  count={1}  class_name={2}'.format(index, count, classes[index]))

    indices = list(range(len(full_set)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(NUM_CLASSES, int(len(indices) * val_ratio))
    val_size = min(val_size, len(indices) - NUM_CLASSES)
    if val_size <= 0:
        raise ValueError('训练图片不足，无法划分验证集。')
    val_indices, train_indices = indices[:val_size], indices[val_size:]
    print('训练图片：{0}，内部验证图片：{1}'.format(len(train_indices), len(val_indices)))
    common = {'batch_size': batch_size, 'num_workers': workers,
              'pin_memory': torch.cuda.is_available()}
    return (
        DataLoader(RemappedSubset(full_set, train_indices, mapping), shuffle=True, **common),
        DataLoader(RemappedSubset(full_set, val_indices, mapping), shuffle=False, **common),
    )


def macro_f1(truths, predictions):
    scores = []
    for index in range(NUM_CLASSES):
        tp = sum(a == index and b == index for a, b in zip(truths, predictions))
        fp = sum(a != index and b == index for a, b in zip(truths, predictions))
        fn = sum(a == index and b != index for a, b in zip(truths, predictions))
        scores.append(0.0 if not (2 * tp + fp + fn) else float(2 * tp) / (2 * tp + fp + fn))
    return sum(scores) / NUM_CLASSES


def evaluate(model, loader, criterion, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    truths, predictions = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predicted = outputs.argmax(dim=1)
            size = targets.size(0)
            total += size
            correct += (predicted == targets).sum().item()
            loss_sum += loss.item() * size
            truths.extend(targets.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
    return loss_sum / total, float(correct) / total, macro_f1(truths, predictions)


def parse_args():
    parser = argparse.ArgumentParser(description='训练 25 类恶劣天气交通标志模型。')
    parser.add_argument('--train-dir', type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0, help='线上环境建议使用 0。')
    parser.add_argument('--device', default='auto', help='auto、cpu、cuda 或 cuda:0。')
    return parser.parse_args()


def train():
    args = parse_args()
    if not 0 < args.val_ratio < 1 or min(args.epochs, args.batch_size, args.image_size) <= 0 or args.num_workers < 0:
        raise ValueError('训练参数不合法。')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not args.train_dir.is_dir():
        raise FileNotFoundError('训练目录不存在：{0}'.format(args.train_dir))
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定了 CUDA，但当前环境没有可用 CUDA。')
    classes = load_classes()
    train_loader, val_loader = build_loaders(args.train_dir, classes, args.image_size,
                                             args.val_ratio, args.batch_size, args.seed, args.num_workers)
    print('device：{0}'.format(device))
    model = TrafficSignCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_f1 = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total += targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            loss_sum += loss.item() * targets.size(0)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print('Epoch {0}/{1}  train_loss={2:.4f}  train_acc={3:.4f}  val_loss={4:.4f}  val_acc={5:.4f}  val_macro_f1={6:.4f}'.format(
            epoch, args.epochs, loss_sum / total, float(correct) / total, val_loss, val_acc, val_f1))
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model_state_dict': model.state_dict(),
                        'class_to_idx': dict((name, i) for i, name in enumerate(classes)),
                        'idx_to_class': dict((i, name) for i, name in enumerate(classes)),
                        'input_size': args.image_size, 'model_name': 'TrafficSignCNN',
                        'num_classes': NUM_CLASSES, 'best_macro_f1': best_f1, 'epoch': epoch}, str(args.output))
            print('已更新最佳 checkpoint：{0}'.format(args.output))


if __name__ == '__main__':
    train()

