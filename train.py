import torch
from torch.utils.data import DataLoader
from data import AudioDataset
from model import AudioClassifier
from sklearn.model_selection import train_test_split
import os

ds = AudioDataset(root_dir="Dataset_")
train_ds, val_ds = train_test_split(ds, test_size=0.2, random_state=42)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = len(os.listdir("Dataset_"))

model = AudioClassifier(n_classes=n_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

criterion = torch.nn.CrossEntropyLoss()

n_epochs = 100

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    train_loss_list = []
    train_acc_list = []

    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_ = model(X)

        loss = criterion(y_, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * X.size(0)

        _, y_label_ = torch.max(y_, 1)

        train_acc += torch.sum(y_label_ == y.data)

        train_loss_list.append(loss.item())
        train_acc_list.append(torch.sum(y_label_ == y.data))
    
    train_loss = train_loss / len(train_ds)

    train_acc = train_acc / len(train_ds)

    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    val_loss_list = []
    val_acc_list = []

    for i, (X, y) in enumerate(val_loader):
        X = X.to(device)
        y = y.to(device)

        y_ = model(X)

        loss = criterion(y_, y)

        val_loss += loss.item() * X.size(0)

        _, y_label_ = torch.max(y_, 1)

        val_acc += torch.sum(y_label_ == y.data)

        val_loss_list.append(loss.item())
        val_acc_list.append(torch.sum(y_label_ == y.data))
    
    val_loss = val_loss / len(val_ds)

    val_acc = val_acc / len(val_ds)

    print(f"Epoch: {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

    torch.save(model.state_dict(), f"checkpoint/model_{epoch+1}.pth")