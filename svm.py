import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ✅ Step 1: 用 h5py 读取 .mat v7.3 文件
with h5py.File('D:/op/new/svmxunlian/GM_Features.mat', 'r') as f:
    X = np.array(f['gm_features']).T  # 注意：转置
    y = np.array(f['labels']).T.ravel()  # 标签拉平为一维

# ✅ Step 2: 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Step 3: 划分训练集与测试集（例如 80% 训练, 20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Step 4: 初始化并训练 SVM 分类器（RBF核）
svm_model = SVC(kernel='rbf', C=1.0, probability=True)
svm_model.fit(X_train, y_train)

# ✅ Step 5: 预测与评估
y_pred = svm_model.predict(X_test)
y_prob = svm_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
conf = confusion_matrix(y_test, y_pred)

print("✅ SVM Accuracy:", accuracy)
print("✅ SVM AUC-ROC:", auc)
print("✅ Confusion Matrix:\n", conf)

# ✅ Step 6: 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label='SVM (AUC = %.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
