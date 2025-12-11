import streamlit as st

st.set_page_config(
    page_title="線上學習能力分析報告",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 關於線上學習能力之預測與關鍵因素分析")
st.markdown("---")

st.header("🚀 應用程式簡介與目的")
st.write("""
**Streamlit 是一個專為機器學習和數據科學專案量身打造的開源應用程式框架。**

本應用程式展示了我們的模型在 **預測個體的線上學習能力** 上的分析成果。我們的核心目標是：
1.  **量化** 個體的學習表現預測值。
2.  **識別** 影響學習能力的關鍵因素（例如：自我效能、學習時長等）。
""")


st.subheader("💡 報告導覽指引")
st.markdown("""
請依序查看以下分析報告，以全面了解模型的表現和結果：

* **a. 特徵分析 (Feature)：** 哪些因素對預測結果影響最大？ 
* **b. 準確度 (Accuracy)：** 模型整體預測的準確程度如何？
* **c. 混淆矩陣 (Confusion Matrix)：** 模型在分類上的具體成功與失敗案例分佈。 
* **d. ROC 曲線 (ROC)：** 評估模型在不同閾值下的分辨能力和穩定性。 
""")

st.markdown("---")
st.caption("數據科學與機器學習應用演示報告。")
