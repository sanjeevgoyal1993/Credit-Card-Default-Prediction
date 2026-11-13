# Credit Card Default Prediction

Predicts which credit card customers will default on their next payment, using 30,000 customers from Taiwan (April to September 2005). The goal is a reliable **probability of default**, not just a yes/no label, so models are judged on ranking power (ROC-AUC, K-S) and on how accurate the probabilities are (Brier score, calibration).

- **Notebook:** [Credit_Card_Default_Prediction.ipynb](Credit_Card_Default_Prediction.ipynb)
- **Colab (view only):** [open the notebook in Google Colab](https://colab.research.google.com/drive/1TYB817cdC2UBxE5TJGYILWHyw5VE3MXS). To run it, use File → Save a copy in Drive, then Runtime → Run all.
- **Author:** Sanjeev Goyal. ML capstone project, AlmaBetter. Developed with AI assistance from Claude.

## Dataset

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI Machine Learning Repository): 30,000 rows and 23 features covering the credit limit, demographics, and six months of repayment status, bill amounts and payments. The notebook reads the course's dataset file from a public link, so it runs end to end without mounting Google Drive.

## Approach

1. **Know the data:** no missing values and no duplicate IDs. Clean-up focuses on undocumented codes.
2. **18 charts following the UBM rule:** univariate, bivariate and multivariate, each answering why it was chosen, what it shows and the business impact.
3. **3 hypothesis tests:** chi-square for education and marital status, and a one-sided Mann-Whitney U test for credit limit, each with an effect size.
4. **Feature engineering:** late months, worst delay, utilisation, repayment ratio and bill trend. Twelve numeric features with a maximum VIF of 4, against 27 for the raw columns. Gender is excluded.
5. **Three models, each tuned with randomized search and cross-validation:** logistic regression, random forest and XGBoost.
6. **Final model:** K-S chart, gains table, calibration, decision thresholds, risk tiers, permutation importance and SHAP.

## Results on the held-out test set

| Model (tuned) | ROC-AUC | K-S | PR-AUC | Brier |
|---|---|---|---|---|
| Logistic regression | 0.752 | 0.395 | 0.509 | 0.142 |
| Random forest | 0.777 | 0.423 | 0.562 | 0.135 |
| **XGBoost (chosen)** | **0.780** | **0.432** | **0.567** | **0.134** |

## Key findings

- **Behaviour predicts default, demographics barely do.** Customers two months late on the latest statement default 69% of the time, against 13% for those with status code 0.
- **Bills look alike, payments do not.** Defaulters owe about the same as non-defaulters but pay back much less.
- **The score concentrates risk.** The riskiest 10% of customers default at 70%, 3.2 times the average, and the riskiest 30% hold 63% of all defaults.
- **The bank can act in tiers.** The two highest-risk tiers are 22.4% of customers but hold 54.8% of defaults.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook Credit_Card_Default_Prediction.ipynb
```

Tested on Python 3.13 with the versions in `requirements.txt`. The saved outputs come from Google Colab (Linux). XGBoost's random sampling differs slightly between operating systems, so a run on Windows gives slightly different XGBoost numbers. A full run takes about 5 minutes in Colab.

## Limitations

The data covers six months in one country in 2005. Some codes are undocumented, the 5-to-1 cost ratio is an assumption, and age, education and marital status need a legal and fairness review before any real use. The model has not been deployed; the saved pipeline makes a Streamlit app or an Azure Machine Learning endpoint a natural next step.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, statsmodels, scikit-learn, XGBoost, SHAP
