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

from pyspark.sql import Window
from pyspark.sql.functions import col, lit, when, datediff
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_features_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_feature_store_directory, spark):

    # Read silver tables
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + ".parquet"
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print("loaded from:", filepath, "row count:", df.count())

    # Enforce snapshot_date schema
    if dict(df.dtypes).get("snapshot_date") == "date":
        df = df.withColumn("snapshot_date", F.date_format(col("snapshot_date"), "yyyy-MM-dd"))

    # replace null or missing values to 0
    num_cols = ["mob","dpd","due_amt","paid_amt","balance","loan_amt",
                "Total_EMI_per_month","Interest_Rate","tenure"]
    fill_map = {c:0 for c in num_cols if c in df.columns}
    if fill_map:
        df = df.fillna(fill_map)
    df = df.fillna({"Customer_ID": "missing"})

    # Calculate maturity_date and days_to_maturity
    if "loan_start_date" in df.columns:
        if dict(df.dtypes).get("loan_start_date") == "string":
            df = df.withColumn("loan_start_date_dt", F.to_date("loan_start_date", "yyyy-MM-dd"))
        else:
            df = df.withColumn("loan_start_date_dt", col("loan_start_date").cast(DateType()))
    else:
        df = df.withColumn("loan_start_date_dt", lit(None).cast(DateType()))

    df = df.withColumn(
        "maturity_date_str",
        F.when(col("loan_start_date_dt").isNotNull() & col("tenure").isNotNull(),
               F.date_format(F.add_months(col("loan_start_date_dt"), col("tenure").cast(IntegerType())), "yyyy-MM-dd")
        ).otherwise(lit(None).cast(StringType()))
    )

    df = df.withColumn(
        "days_to_maturity",
        when(
            F.to_date("maturity_date_str","yyyy-MM-dd").isNotNull() & F.to_date("snapshot_date","yyyy-MM-dd").isNotNull(),
            datediff(F.to_date("maturity_date_str","yyyy-MM-dd"), F.to_date("snapshot_date","yyyy-MM-dd"))
        ).otherwise(lit(None).cast(IntegerType()))
    )

    # Credit_Utilization_Ratio
    if "Credit_Utilization_Ratio" in df.columns:
        df = df.withColumn("utilization", col("Credit_Utilization_Ratio").cast(FloatType()))
    else:
        df = df.withColumn(
            "utilization",
            when(col("loan_amt") > 0, (col("balance")/col("loan_amt")).cast(FloatType())).otherwise(lit(0.0))
        )

    # installment_amt
    if "Total_EMI_per_month" in df.columns:
        df = df.withColumn("installment_amt", col("Total_EMI_per_month").cast(FloatType()))
    else:
        df = df.withColumn("installment_amt", lit(None).cast(FloatType()))

    # interest_rate
    if "Interest_Rate" in df.columns:
        df = df.withColumn("interest_rate", col("Interest_Rate").cast(FloatType()))
    else:
        df = df.withColumn("interest_rate", lit(None).cast(FloatType()))

    # Pay ratio
    df = df.withColumn(
        "pay_ratio",
        when(col("due_amt") > 0, (col("paid_amt")/col("due_amt")).cast(FloatType())).otherwise(lit(None).cast(FloatType()))
    )

    # Current overdue
    df = df.withColumn("flag_30dpd_today", (col("dpd") >= 30).cast(IntegerType())) \
           .withColumn("flag_60dpd_today", (col("dpd") >= 60).cast(IntegerType()))

    # Partition by customer_ID
    cust_win_same_day = Window.partitionBy("Customer_ID","snapshot_date")
    df = df.withColumn("cust_loan_cnt", F.count("*").over(cust_win_same_day)) \
           .withColumn("cust_total_balance", F.sum("balance").over(cust_win_same_day)) \
           .withColumn("cust_max_dpd_today", F.max("dpd").over(cust_win_same_day)) \
           .withColumn("cust_cnt_30dpd_today", F.sum("flag_30dpd_today").over(cust_win_same_day))

    # Select output
    df = df.select(
        "loan_id","Customer_ID","snapshot_date",
        "mob","dpd","loan_amt","balance","due_amt","paid_amt",
        "installment_amt","interest_rate","utilization","pay_ratio","days_to_maturity",
        "flag_30dpd_today","flag_60dpd_today",
        "cust_loan_cnt","cust_total_balance","cust_max_dpd_today","cust_cnt_30dpd_today",
        "loan_start_date","tenure","maturity_date_str"
    )

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + ".parquet"
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df

