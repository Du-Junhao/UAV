import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. 读取数据
data = pd.read_csv("training_data.csv")
X = data.drop(columns=["action"]).values   # 状态向量
y = data["action"].values                  # 标签（0=保持，1=减速，2=加速）

# 2. 特征标准化（非常重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. 定义神经网络模型（MLP）
model = MLPClassifier(
    hidden_layer_sizes=(32, 32),  # 两层，每层32个神经元
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

# 5. 训练模型
model.fit(X_train, y_train)

# 6. 验证准确率
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"✅ 训练完成，验证准确率: {acc:.4f}")

# 7. 保存模型和 scaler
joblib.dump(model, "sk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ 模型已保存为 sk_model.pkl 和 scaler.pkl")
