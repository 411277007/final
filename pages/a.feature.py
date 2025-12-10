import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# Streamlit 標題
st.title("Feature Importance: XGBoost and Random Forest")

# 讀取資料

file_path = r"C:\Users\tw090\OneDrive\桌面\教育大數據系統demo\fixer210_edudata.xlsx"
try:
    df = pd.read_excel(file_path, engine='openpyxl',dtype=float)
    st.write("### 資料預覽")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error(f"找不到檔案：{file_path}")
    st.stop()

# 特徵與目標

feature_cols = df.columns[0:13]  
target_col = df.columns[13]    
X = df[feature_cols]
y = df[target_col]

# 建立模型

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


#  訓練模型

xgb_model.fit(X, y)
rf_model.fit(X, y)

#  取得特徵重要性

def get_importance_df(model, feature_cols):
    importance = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    df_imp['Importance(%)'] = (df_imp['Importance'] * 100).round(2)
    return df_imp

xgb_importance_df = get_importance_df(xgb_model, feature_cols)
rf_importance_df = get_importance_df(rf_model, feature_cols)


#  在 Streamlit 顯示表格

st.header(" XGBoost 特徵重要性")
st.dataframe(xgb_importance_df)

st.header("Random Forest 特徵重要性")
st.dataframe(rf_importance_df)


# 畫圖

def plot_importance(df_imp, title, color):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(df_imp['Feature'], df_imp['Importance'], color=color)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    st.pyplot(fig)

st.header("XGBoost 特徵重要性")
plot_importance(xgb_importance_df, "XGBoost Feature Importance", "orange")

st.header(" Random Forest 特徵重要性")
plot_importance(rf_importance_df, "Random Forest Feature Importance", "skyblue")
