import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.title("六模型混淆矩陣")

DATA_PATH = "new_fixer210_edudata.csv"
TARGET = "Adaptivity Level"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

model_names = list(models.keys())

rows = 3   
cols_per_row = 2  
index = 0

for row in range(rows):
    cols = st.columns(cols_per_row) 
    
    for col in cols:
        if index >= len(model_names):
            break

        model_name = model_names[index]
        model = models[model_name]

        if model_name in ["SVM","ANN"]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)


        with col:
          
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{model_name}")
            st.pyplot(fig)

        index += 1
