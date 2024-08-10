import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

voting_clf_loaded = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

def make_prediction():
    try:
        data = {
            'credit.policy': int(entry_credit_policy.get()),
            'int.rate': float(entry_int_rate.get()),
            'installment': float(entry_installment.get()),
            'log.annual.inc': float(entry_log_annual_inc.get()),
            'dti': float(entry_dti.get()),
            'fico': int(entry_fico.get()),
            'days.with.cr.line': float(entry_days_with_cr_line.get()),
            'revol.bal': float(entry_revol_bal.get()),
            'revol.util': float(entry_revol_util.get()),
            'inq.last.6mths': int(entry_inq_last_6mths.get()),
            'delinq.2yrs': int(entry_delinq_2yrs.get()),
            'pub.rec': int(entry_pub_rec.get()),
            'TotalIncome': float(entry_total_income.get()),
            'purpose_all_other': int(entry_purpose_all_other.get()),
            'purpose_credit_card': int(entry_purpose_credit_card.get()),
            'purpose_debt_consolidation': int(entry_purpose_debt_consolidation.get()),
            'purpose_educational': int(entry_purpose_educational.get()),
            'purpose_home_improvement': int(entry_purpose_home_improvement.get()),
            'purpose_major_purchase': int(entry_purpose_major_purchase.get()),
            'purpose_small_business': int(entry_purpose_small_business.get())
        }

        new_data = pd.DataFrame([data])

        numerical_cols = ['int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line',
                          'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs',
                          'pub.rec', 'TotalIncome']

        new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

        features_for_prediction = new_data
        prediction = voting_clf_loaded.predict(features_for_prediction)

        result = "Prediction: " + ("Approved" if prediction[0] == 1 else "Denied")
        messagebox.showinfo("Result", result)

    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Loan Prediction")

tk.Label(root, text="credit.policy:").grid(row=0, column=0)
entry_credit_policy = tk.Entry(root)
entry_credit_policy.grid(row=0, column=1)

tk.Label(root, text="int.rate:").grid(row=1, column=0)
entry_int_rate = tk.Entry(root)
entry_int_rate.grid(row=1, column=1)

tk.Label(root, text="installment:").grid(row=2, column=0)
entry_installment = tk.Entry(root)
entry_installment.grid(row=2, column=1)

tk.Label(root, text="log.annual.inc:").grid(row=3, column=0)
entry_log_annual_inc = tk.Entry(root)
entry_log_annual_inc.grid(row=3, column=1)

tk.Label(root, text="dti:").grid(row=4, column=0)
entry_dti = tk.Entry(root)
entry_dti.grid(row=4, column=1)

tk.Label(root, text="fico:").grid(row=5, column=0)
entry_fico = tk.Entry(root)
entry_fico.grid(row=5, column=1)

tk.Label(root, text="days.with.cr.line:").grid(row=6, column=0)
entry_days_with_cr_line = tk.Entry(root)
entry_days_with_cr_line.grid(row=6, column=1)

tk.Label(root, text="revol.bal:").grid(row=7, column=0)
entry_revol_bal = tk.Entry(root)
entry_revol_bal.grid(row=7, column=1)

tk.Label(root, text="revol.util:").grid(row=8, column=0)
entry_revol_util = tk.Entry(root)
entry_revol_util.grid(row=8, column=1)

tk.Label(root, text="inq.last.6mths:").grid(row=9, column=0)
entry_inq_last_6mths = tk.Entry(root)
entry_inq_last_6mths.grid(row=9, column=1)

tk.Label(root, text="delinq.2yrs:").grid(row=10, column=0)
entry_delinq_2yrs = tk.Entry(root)
entry_delinq_2yrs.grid(row=10, column=1)

tk.Label(root, text="pub.rec:").grid(row=11, column=0)
entry_pub_rec = tk.Entry(root)
entry_pub_rec.grid(row=11, column=1)

tk.Label(root, text="TotalIncome:").grid(row=12, column=0)
entry_total_income = tk.Entry(root)
entry_total_income.grid(row=12, column=1)

tk.Label(root, text="purpose_all_other:").grid(row=13, column=0)
entry_purpose_all_other = tk.Entry(root)
entry_purpose_all_other.grid(row=13, column=1)

tk.Label(root, text="purpose_credit_card:").grid(row=14, column=0)
entry_purpose_credit_card = tk.Entry(root)
entry_purpose_credit_card.grid(row=14, column=1)

tk.Label(root, text="purpose_debt_consolidation:").grid(row=15, column=0)
entry_purpose_debt_consolidation = tk.Entry(root)
entry_purpose_debt_consolidation.grid(row=15, column=1)

tk.Label(root, text="purpose_educational:").grid(row=16, column=0)
entry_purpose_educational = tk.Entry(root)
entry_purpose_educational.grid(row=16, column=1)

tk.Label(root, text="purpose_home_improvement:").grid(row=17, column=0)
entry_purpose_home_improvement = tk.Entry(root)
entry_purpose_home_improvement.grid(row=17, column=1)

tk.Label(root, text="purpose_major_purchase:").grid(row=18, column=0)
entry_purpose_major_purchase = tk.Entry(root)
entry_purpose_major_purchase.grid(row=18, column=1)

tk.Label(root, text="purpose_small_business:").grid(row=19, column=0)
entry_purpose_small_business = tk.Entry(root)
entry_purpose_small_business.grid(row=19, column=1)

tk.Button(root, text="Predict", command=make_prediction).grid(row=20, column=1)

root.mainloop()
