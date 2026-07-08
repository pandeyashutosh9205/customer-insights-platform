# 🛍️ Intelligent Customer Insights & Support Platform

*What if one dashboard could tell you who's about to churn, how much you'll sell next week, and what your customers are actually feeling — all at once?*

That's what this is. A single Streamlit app built on 100K+ real e-commerce orders ([Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)) that combines EDA, machine learning, deep learning, and NLP into four working tools.

---

## 🔍 What it does

| Module | Question it answers | How |
|---|---|---|
| **Insights Dashboard** | What's happening with our sales? | EDA + RFM customer segmentation |
| **Churn Risk Checker** | Who's about to leave? | Random Forest classifier (ROC-AUC 0.65) |
| **Sales Forecast** | What will we sell tomorrow? | LSTM neural network (MAE ≈ R$7,675) |
| **Review Analyzer** | What are customers feeling? | TF-IDF + Logistic Regression (77% accuracy) |

---

## 🐛 The bug that taught me the most

My first churn model scored a *perfect* 1.00 ROC-AUC. Exciting — until I realized it was cheating: I'd accidentally fed it the exact feature (`recency`) I'd used to define the churn label in the first place. It wasn't predicting churn, it was just reading the answer key.

Fixing it dropped the score to a far more honest **0.65** — a smaller number, but a real one. This is the version in the app.

---

## 🛠️ Built with

Python · Pandas · scikit-learn · TensorFlow/Keras · Streamlit

---

## ▶️ Run it yourself

```bash
git clone https://github.com/pandeyashutosh9205/customer-insights-platform.git
cd customer-insights-platform
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 🙋 Author

**Ashutosh Pandey** — [GitHub](https://github.com/pandeyashutosh9205) · [LinkedIn](https://www.linkedin.com/in/ashutosh-pandey-3b323b320/)