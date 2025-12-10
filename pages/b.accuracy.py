import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

st.title(" 多模型分類結果比較 ")


DATA_PATH = "new_fixer210_edudata.csv"       
TARGET = "Adaptivity Level"     

# 讀取資料
df = pd.read_csv(DATA_PATH)

st.write(f"資料檔案：**{DATA_PATH}**")
st.write(f"預測欄位：**{TARGET}**")


# 資料分割

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 定義模型

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, use_label_encoder=False, eval_metric='mlogloss'
    ),

    "SVM": SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naive Bayes": GaussianNB(),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), 
        activation='relu', 
        solver='adam', 
        max_iter=1000, 
        random_state=42,
        early_stopping=True),
    }



# 訓練 & 計算指標

results = []

for name, model in models.items():
    model.fit(X_train, y_train)         # 訓練模型
    y_pred = model.predict(X_test)      # 預測
    
    accuracy = accuracy_score(y_test, y_pred) * 100   # 百分比
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_avg = report["macro avg"]["f1-score"]
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        "model": name,
        "accuracy (%)": f"{accuracy:.2f}%",
        "macro avg": round(macro_avg, 4),
        "f1 score": round(f1, 4)
    })


# 顯示表格

results_df = pd.DataFrame(results)
st.subheader("模型指標比較表")
st.dataframe(results_df)
