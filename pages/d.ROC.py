import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
st.title("六模型 Macro-Average ROC 曲線比較")

DATA_PATH = "new_fixer210_edudata.csv"
TARGET = "Adaptivity Level"

df = pd.read_csv(DATA_PATH)
selected_features = [
    'Class Duration', 'Financial Condition', 'Age', 'Gender', 
    'Network Type', 'Institution Type', 'Education Level', 
    'Location', 'Internet Type', 'Self Lms', 'IT Student', 'Load-shedding'
]

X = df[selected_features]
y = df[TARGET]

y_encoded = y.values     
classes = sorted(y.unique())
n_classes = len(classes)

from sklearn.preprocessing import label_binarize
y_bin = label_binarize(y_encoded, classes=classes)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_bin, test_size=0.2, random_state=42, stratify=y_encoded
)


models = [
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("XGBoost", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ("SVM", OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=42))),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("Naive Bayes", OneVsRestClassifier(GaussianNB())),
    ("ANN (MLP)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
]


plt.figure(figsize=(10, 8))
st.write("訓練各模型中...")

for name, model in models:
   
    classifier = model if isinstance(model, OneVsRestClassifier) else OneVsRestClassifier(model)
    classifier.fit(X_train, y_train)

    # 取得預測機率
    if hasattr(classifier, "predict_proba"):
        y_score = classifier.predict_proba(X_test)
    else:
        y_score = classifier.decision_function(X_test)

    # 計算每個類別的 FPR / TPR / AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 畫 ROC
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'{name} (AUC = {roc_auc["macro"]:.2f})',
             linewidth=2)
    st.write(f"✅ {name} 完成！Macro-AUC: {roc_auc['macro']:.4f}")


# 美化圖表

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC=0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-Average ROC ', fontsize=16)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
st.pyplot(plt)
