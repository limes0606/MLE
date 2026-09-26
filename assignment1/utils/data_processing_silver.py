"""
Silver layer processing.

Silver = cleaned, conformed, typed data at the same grain as bronze, with
malformed/placeholder values fixed or nulled out.
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType


def _write_silver(df, table_name, snapshot_date, silver_dir):
    out_dir = os.path.join(silver_dir, table_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"silver_{table_name}_{snapshot_date.replace('-', '_')}.parquet")
    df.write.mode("overwrite").parquet(out_path)
    print(f"[silver] {table_name:12s} {snapshot_date}: {df.count():>6d} rows -> {out_path}")
    return df


def _strip_to_number(colname):
    col = F.col(colname).cast("string")
    candidate = F.regexp_replace(col, r"^_+|_+$", "")
    return F.when(candidate.cast(DoubleType()).isNotNull(), candidate).otherwise(None)


def process_silver_attributes(bronze_df, snapshot_date, silver_dir):
    df = bronze_df
    df = df.withColumn("Age", _strip_to_number("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((F.col("Age") < 18) | (F.col("Age") > 100), None).otherwise(F.col("Age")))
    df = df.withColumn(
        "Occupation",
        F.when(F.col("Occupation").rlike(r"^[A-Za-z_ ]+$") & (F.col("Occupation") != "_______"), F.col("Occupation")).otherwise(None),
    )
    return _write_silver(df, "attributes", snapshot_date, silver_dir)


NUMERIC_FINANCIAL_COLS = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance",
]


def process_silver_financials(bronze_df, snapshot_date, silver_dir):
    df = bronze_df
    for c in NUMERIC_FINANCIAL_COLS:
        df = df.withColumn(c, _strip_to_number(c).cast(DoubleType()))

    df = df.withColumn(
        "Annual_Income",
        F.when(F.col("Annual_Income") > 1000000, None).otherwise(F.col("Annual_Income")),
    )
    df = df.withColumn(
        "Num_of_Loan",
        F.when((F.col("Num_of_Loan") < 0) | (F.col("Num_of_Loan") > 20), None).otherwise(F.col("Num_of_Loan")).cast(IntegerType()),
    )
    df = df.withColumn(
        "Interest_Rate",
        F.when((F.col("Interest_Rate") < 0) | (F.col("Interest_Rate") > 100), None).otherwise(F.col("Interest_Rate")),
    )
    df = df.withColumn(
        "Num_Bank_Accounts",
        F.when((F.col("Num_Bank_Accounts") < 0) | (F.col("Num_Bank_Accounts") > 20), None).otherwise(F.col("Num_Bank_Accounts")),
    )
    df = df.withColumn(
        "Num_Credit_Card",
        F.when((F.col("Num_Credit_Card") < 0) | (F.col("Num_Credit_Card") > 20), None).otherwise(F.col("Num_Credit_Card")),
    )
    df = df.withColumn(
        "Num_Credit_Inquiries",
        F.when((F.col("Num_Credit_Inquiries") < 0) | (F.col("Num_Credit_Inquiries") > 50), None).otherwise(F.col("Num_Credit_Inquiries")),
    )
    df = df.withColumn(
        "Num_of_Delayed_Payment",
        F.when((F.col("Num_of_Delayed_Payment") < 0) | (F.col("Num_of_Delayed_Payment") > 50), None).otherwise(F.col("Num_of_Delayed_Payment")),
    )
    df = df.withColumn(
        "Delay_from_due_date",
        F.when(F.col("Delay_from_due_date") < 0, None).otherwise(F.col("Delay_from_due_date")),
    )
    df = df.withColumn(
        "Monthly_Balance",
        F.when(F.col("Monthly_Balance") < 0, None).otherwise(F.col("Monthly_Balance")),
    )
    df = df.withColumn(
        "Total_EMI_per_month",
        F.when(F.col("Total_EMI_per_month") < 0, None).otherwise(F.col("Total_EMI_per_month")),
    )

    df = df.withColumn(
        "Credit_Mix",
        F.when(F.col("Credit_Mix").isin("Good", "Standard", "Bad"), F.col("Credit_Mix")).otherwise(None),
    )
    df = df.withColumn(
        "Payment_of_Min_Amount",
        F.when(F.col("Payment_of_Min_Amount").isin("Yes", "No"), F.col("Payment_of_Min_Amount")).otherwise(None),
    )
    df = df.withColumn(
        "Payment_Behaviour",
        F.when(
            F.col("Payment_Behaviour").rlike(r"^(Low|High)_spent_(Small|Medium|Large)_value_payments$"),
            F.col("Payment_Behaviour"),
        ).otherwise(None),
    )

    # Credit_History_Age arrives as free text, e.g. "10 Years and 9 Months".
    # Parse into a single numeric column: total months of credit history.
    years = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s*Years?", 1)
    months = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s*Months?", 1)
    df = df.withColumn(
        "Credit_History_Age_Months",
        F.when(
            F.col("Credit_History_Age").isNotNull() & (years != "") & (months != ""),
            years.cast(IntegerType()) * 12 + months.cast(IntegerType()),
        ).otherwise(None),
    ).drop("Credit_History_Age")

    # Type_of_Loan arrives as a comma-separated free-text list, e.g.
    # "Auto Loan, Credit-Builder Loan, Personal Loan". Parse into a count
    # of distinct loan types rather than passing the raw string through.
    df = df.withColumn(
        "Num_Type_of_Loan",
        F.when(
            F.col("Type_of_Loan").isNotNull()
            & F.col("Type_of_Loan").rlike(r"^[A-Za-z\- ]+(,\s*[A-Za-z\- ]+)*$")
            & (F.col("Type_of_Loan") != "Not Specified"),
            F.size(F.split(F.col("Type_of_Loan"), r",\s*")),
        ).otherwise(None),
    ).drop("Type_of_Loan")

    return _write_silver(df, "financials", snapshot_date, silver_dir)


def process_silver_loan_daily(bronze_df, snapshot_date, silver_dir):
    df = bronze_df
    df = (
        df.withColumn("loan_id", F.col("loan_id").cast("string"))
          .withColumn("Customer_ID", F.col("Customer_ID").cast("string"))
          .withColumn("loan_start_date", F.to_date("loan_start_date"))
          .withColumn("snapshot_date", F.to_date("snapshot_date"))
          .withColumn("tenure", F.col("tenure").cast(IntegerType()))
          .withColumn("installment_num", F.col("installment_num").cast(IntegerType()))
    )
    for c in ["loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance"]:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))
    return _write_silver(df, "loan_daily", snapshot_date, silver_dir)


def process_silver_clickstream(bronze_df, snapshot_date, silver_dir):
    df = bronze_df.withColumn("Customer_ID", F.col("Customer_ID").cast("string"))
    df = df.withColumn("snapshot_date", F.to_date("snapshot_date"))
    return _write_silver(df, "clickstream", snapshot_date, silver_dir)


SILVER_PROCESSORS = {
    "attributes": process_silver_attributes,
    "financials": process_silver_financials,
    "loan_daily": process_silver_loan_daily,
    "clickstream": process_silver_clickstream,
}