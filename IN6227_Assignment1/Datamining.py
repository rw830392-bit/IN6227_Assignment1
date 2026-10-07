import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. 确保输出文件夹存在
os.makedirs('outputs', exist_ok=True)

# 2. 读取数据集
train_path = 'dataset/train.csv'
test_path = 'dataset/test.csv'

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("未找到数据集文件，请确认 dataset 目录下包含 train.csv 和 test.csv！")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"=== 数据成功加载 ===")
print(f"训练集原始维度: {train_df.shape}")
print(f"测试集原始维度: {test_df.shape}\n")

# 3. 动态获取目标列名，并清理目标列中的 NaN 行
TARGET_COL = train_df.columns[-1]

# 检查并剔除目标列中包含 NaN 的样本行
nan_count = train_df[TARGET_COL].isna().sum()
if nan_count > 0:
    print(f"[提示] 检测到目标列 '{TARGET_COL}' 包含 {nan_count} 个缺失值，已自动删除这些无标签样本。")
    train_df = train_df.dropna(subset=[TARGET_COL])
    print(f"清理后训练集维度: {train_df.shape}\n")

print(f"目标变量列名 (Target Column): '{TARGET_COL}'")
print(f"目标类别比例分布:\n{train_df[TARGET_COL].value_counts(normalize=True)}\n")

X = train_df.drop(columns=[TARGET_COL])
y = train_df[TARGET_COL]

# 4. 识别特征类型（更加兼容的数值/类别特征提取方式，消除警告）
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"数值特征 ({len(num_features)} 个): {num_features}")
print(f"类别特征 ({len(cat_features)} 个): {cat_features}\n")

# 5. 构建预处理 Pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 中位数填补缺失值
    ('scaler', StandardScaler())  # 标准化特征缩放
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 众数填补缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 独热编码
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# 划分训练集与验证集 (80% 训练, 20% 验证, 分层抽样)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 超参数寻优与模型训练 (5-Fold Stratified CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 模型 A: 逻辑回归 (Logistic Regression)
pipe_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

param_lr = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__solver': ['lbfgs']
}

grid_lr = GridSearchCV(pipe_lr, param_lr, cv=cv, scoring='f1_weighted', n_jobs=-1)
grid_lr.fit(X_train, y_train)

# 模型 B: 随机森林 (Random Forest)
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(pipe_rf, param_rf, cv=cv, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train, y_train)


# 7. 模型评估指标计算
def evaluate_model(model, X_v, y_v, name):
    preds = model.predict(X_v)

    # 尝试获取概率输出以计算 AUC
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(X_v)
        # 将字符串标签匹配转换为索引计算 AUC
        classes = list(model.classes_)
        positive_class = classes[1] if len(classes) == 2 else None
        proba = proba_all[:, 1] if positive_class else None
    else:
        proba = None

    acc = accuracy_score(y_v, preds)
    prec = precision_score(y_v, preds, average='weighted', zero_division=0)
    rec = recall_score(y_v, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_v, preds, average='weighted', zero_division=0)

    is_binary = len(np.unique(y_v)) == 2
    auc = roc_auc_score(y_v, proba) if proba is not None and is_binary else np.nan

    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }, preds, proba


metrics_lr, preds_lr, proba_lr = evaluate_model(grid_lr.best_estimator_, X_val, y_val, "Logistic Regression")
metrics_rf, preds_rf, proba_rf = evaluate_model(grid_rf.best_estimator_, X_val, y_val, "Random Forest")

# 打印对比结果表
df_results = pd.DataFrame([metrics_lr, metrics_rf])
print("=== 验证集模型性能对比 ===")
print(df_results.to_string(index=False))
print("\nLogistic Regression 最佳参数:", grid_lr.best_params_)
print("Random Forest 最佳参数:", grid_rf.best_params_)

# 8. 图表绘制与保存 (供报告插入使用)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# (1) ROC 曲线对比
ax1 = axes[0]
is_binary = len(np.unique(y_val)) == 2
if proba_lr is not None and proba_rf is not None and is_binary:
    # 适配分类标签为非数字类型（如 'yes'/'no'）的情况
    pos_label = grid_rf.best_estimator_.classes_[1]
    fpr_lr, tpr_lr, _ = roc_curve(y_val, proba_lr, pos_label=pos_label)
    fpr_rf, tpr_rf, _ = roc_curve(y_val, proba_rf, pos_label=pos_label)

    ax1.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {metrics_lr["ROC-AUC"]:.3f})', color='#1f77b4', lw=2)
    ax1.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {metrics_rf["ROC-AUC"]:.3f})', color='#ff7f0e', lw=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
else:
    ax1.text(0.5, 0.5, 'ROC Curve\n(Multi-class or No Proba)', ha='center', va='center', fontsize=12)

# (2) 最佳模型混淆矩阵
ax2 = axes[1]
best_preds = preds_rf if metrics_rf['F1-Score'] >= metrics_lr['F1-Score'] else preds_lr
best_name = "Random Forest" if metrics_rf['F1-Score'] >= metrics_lr['F1-Score'] else "Logistic Regression"

cm = confusion_matrix(y_val, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title(f'Confusion Matrix ({best_name})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted Label', fontsize=11)
ax2.set_ylabel('True Label', fontsize=11)

plt.tight_layout()
chart_path = 'outputs/model_evaluation_charts.png'
plt.savefig(chart_path, bbox_inches='tight')
plt.close()
print(f"\n[成功] 评估图表已导出至: {chart_path}")

# 9. 对测试集 test.csv 进行预测并保存
best_model = grid_rf.best_estimator_ if metrics_rf['F1-Score'] >= metrics_lr['F1-Score'] else grid_lr.best_estimator_

X_test = test_df.drop(columns=[TARGET_COL]) if TARGET_COL in test_df.columns else test_df.copy()
test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    'id': test_df.index,
    'prediction': test_preds
})
pred_path = 'outputs/test_predictions.csv'
submission.to_csv(pred_path, index=False)
print(f"[成功] 测试集预测结果已保存至: {pred_path}")

