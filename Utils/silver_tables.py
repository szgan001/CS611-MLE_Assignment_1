import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    token = snapshot_date_str.replace('-', '_')

    loan_fp = os.path.join(bronze_lms_directory, f"bronze_loan_daily_{token}.csv")
    df = spark.read.csv(loan_fp, header=True, inferSchema=True)
    print('loaded from:', loan_fp, 'row count:', df.count())

    # adjust schema
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for c, t in column_type_map.items():
        df = df.withColumn(c, col(c).cast(t))

    # adding variables
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "installments_missed",
        F.when(col("due_amt") > 0, F.ceil(col("overdue_amt") / col("due_amt")))
         .otherwise(0).cast(IntegerType())
    )
    df = df.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0,
               F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
        ).cast(DateType())
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0,
               F.datediff(col("snapshot_date"), col("first_missed_date"))
        ).otherwise(0).cast(IntegerType())
    )

    # Join tables
    ## financials
    fin_fp = os.path.join(bronze_lms_directory, f"bronze_financials_{token}.csv")
    fin_raw = spark.read.csv(fin_fp, header=True, inferSchema=True)
    if "snapshot_date" in fin_raw.columns:
        fin = (fin_raw
               .withColumn("snapshot_date", F.to_date(col("snapshot_date")))
               .filter(col("snapshot_date") == F.to_date(F.lit(snapshot_date_str)))
               .drop("snapshot_date"))
    else:
        fin = fin_raw.dropDuplicates(["Customer_ID"])
    print("financials rows (prepped):", fin.count())

    ## Attributes
    attr_fp = os.path.join(bronze_lms_directory, f"bronze_attributes_{token}.csv")
    attr_raw = spark.read.csv(attr_fp, header=True, inferSchema=True)
    if "snapshot_date" in attr_raw.columns:
        attr = (attr_raw
                .withColumn("snapshot_date", F.to_date(col("snapshot_date")))
                .filter(col("snapshot_date") == F.to_date(F.lit(snapshot_date_str)))
                .drop("snapshot_date"))
    else:
        attr = attr_raw.dropDuplicates(["Customer_ID"])
    print("attributes rows (prepped):", attr.count())

    ## clickstream
    clk_fp = os.path.join(bronze_lms_directory, f"bronze_clickstream_{token}.csv")
    clk_raw = (spark.read.csv(clk_fp, header=True, inferSchema=True)
                    .withColumn("snapshot_date", F.to_date(col("snapshot_date"))))
    clk_raw = clk_raw.filter(col("snapshot_date") == F.to_date(F.lit(snapshot_date_str)))

    fe_cols = [c for c in clk_raw.columns if c.startswith("fe_")]
    if fe_cols:
        agg_exprs = [F.avg(col(c)).alias(c) for c in fe_cols]
        clk = clk_raw.groupBy("Customer_ID", "snapshot_date").agg(*agg_exprs)
    else:
        clk = clk_raw.dropDuplicates(["Customer_ID", "snapshot_date"])
    print("clickstream rows (grouped):", clk.count())

    # Join
    before = df.count()
    df = df.join(fin, on="Customer_ID", how="left")
    after_fin = df.count()

    df = df.join(attr, on="Customer_ID", how="left")
    after_attr = df.count()

    df = df.join(clk, on=["Customer_ID", "snapshot_date"], how="left")
    after_clk = df.count()

    print(f"row counts -> loan:{before}  +fin:{after_fin}  +attr:{after_attr}  +clk:{after_clk}")

    # enforce schema
    target_type_map = {
        # keys / categories
        "loan_id":"string", "Customer_ID":"string", "Type_of_Loan":"string",
        "Credit_Mix":"string", "Payment_of_Min_Amount":"string", "Payment_Behaviour":"string",
        "Name":"string", "SSN":"string", "Occupation":"string",

        # dates
        "snapshot_date":"date", "loan_start_date":"date", "first_missed_date":"date",

        # integers
        "tenure":"int", "installment_num":"int", "mob":"int", "installments_missed":"int",
        "dpd":"int", "Age":"int", "Num_Bank_Accounts":"int", "Num_Credit_Card":"int",
        "Num_of_Loan":"int", "Num_of_Delayed_Payment":"int", "Num_Credit_Inquiries":"int",

        # doubles (amounts/ratios)
        "loan_amt":"double", "due_amt":"double", "paid_amt":"double", "overdue_amt":"double",
        "balance":"double", "Annual_Income":"double", "Monthly_Inhand_Salary":"double",
        "Interest_Rate":"double", "Delay_from_due_date":"double", "Changed_Credit_Limit":"double",
        "Outstanding_Debt":"double", "Credit_Utilization_Ratio":"double",
        "Credit_History_Age":"double", "Total_EMI_per_month":"double",
        "Amount_invested_monthly":"double", "Monthly_Balance":"double",
    }
    for c, t in target_type_map.items():
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(t))

    for c in [c for c in df.columns if c.startswith("fe_")]:
        df = df.withColumn(c, col(c).cast("double"))

    if {"loan_id","snapshot_date"}.issubset(set(df.columns)):
        df = df.dropDuplicates(["loan_id","snapshot_date"])


    # save silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df
