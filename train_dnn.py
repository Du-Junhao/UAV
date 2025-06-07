import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
data = pd.read_csv("training_data.csv")
X = data.drop(columns=["action"]).values   # 状态向量
y = data["action"].values                  # 标签

# 标准化特征（非常重要）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转为张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 2. 划分训练集/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义模型
class DroneDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出3类：0、1、2
        )

    def forward(self, x):
        return self.net(x)

model = DroneDNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    # Verify accuracy
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_pred = val_output.argmax(dim=1)
        acc = (val_pred == y_val).float().mean()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {acc:.4f}")
# 5. Save model and standardizer
torch.save(model.state_dict(), "dnn_model.pth")
import joblib
joblib.dump(scaler, "scaler.pkl")

print("✅ 模型训练完成，已保存为 dnn_model.pth 和 scaler.pkl")
