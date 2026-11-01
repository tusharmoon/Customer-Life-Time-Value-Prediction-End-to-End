import pandas as pd
import numpy as np

def generate_clv_transactions(n_customers=1500, output_file="data/sample_clv_transactions.csv"):
    """
    Generates realistic e-commerce transaction data optimized for both
    RFM segmentation and 12-Month Supervised CLV regression modeling.
    """
    np.random.seed(42)
    customer_ids = [f"CUST_{1000 + i}" for i in range(n_customers)]
    weights = [0.10, 0.30, 0.30, 0.30]
    cust_types = np.random.choice(['vip', 'loyal', 'growth', 'at_risk'], size=n_customers, p=weights)

    transactions = []
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2026-03-31')
    date_range_days = (end_date - start_date).days

    for i, cid in enumerate(customer_ids):
        ctype = cust_types[i]
        if ctype == 'vip':
            t_count = np.random.randint(12, 35)
            spend_mu, spend_sd = 250, 80
            recency_max_days = 60
        elif ctype == 'loyal':
            t_count = np.random.randint(5, 15)
            spend_mu, spend_sd = 120, 40
            recency_max_days = 90
        elif ctype == 'growth':
            t_count = np.random.randint(2, 6)
            spend_mu, spend_sd = 85, 30
            recency_max_days = 120
        else:
            t_count = np.random.randint(1, 3)
            spend_mu, spend_sd = 45, 20
            recency_max_days = 700

        last_tx_day = date_range_days - np.random.randint(1, recency_max_days)
        if last_tx_day < 30:
            last_tx_day = 30
        
        span = np.random.randint(1, last_tx_day)
        tx_days = np.sort(np.random.choice(range(last_tx_day - span, last_tx_day + 1), size=t_count, replace=True))
        
        for day_offset in tx_days:
            tx_date = start_date + pd.Timedelta(days=int(day_offset))
            amount = round(max(10.0, np.random.normal(spend_mu, spend_sd)), 2)
            invoice_id = f"INV-{np.random.randint(100000, 999999)}"
            transactions.append({
                "InvoiceNo": invoice_id,
                "CustomerID": cid,
                "TransactionDate": tx_date.strftime("%Y-%m-%d"),
                "Amount": amount
            })

    df = pd.DataFrame(transactions)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} transactions for {n_customers} distinct customers in '{output_file}'.")

if __name__ == "__main__":
    generate_clv_transactions()
