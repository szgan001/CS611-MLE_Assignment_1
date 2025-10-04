import os
from pyspark.sql.functions import col

def process_bronze_tables(snapshot_date_str, bronze_lms_directory, spark):
    datasets = [
        ("data/lms_loan_daily.csv",      "bronze_loan_daily"),
        ("data/feature_clickstream.csv",  "bronze_clickstream"),
        ("data/features_attributes.csv", "bronze_attributes"),
        ("data/features_financials.csv",   "bronze_financials"),
    ]
    
    results = {}
    for csv_file_path, prefix in datasets:
        df = (spark.read.csv(csv_file_path, header=True, inferSchema=True)
                     .filter(col("snapshot_date") == snapshot_date_str))

        print(prefix, snapshot_date_str, "row count:", df.count())

        partition_name = f"{prefix}_{snapshot_date_str.replace('-', '_')}.csv"
        filepath = os.path.join(bronze_lms_directory, partition_name)

        df.toPandas().to_csv(filepath, index=False)
        print("saved to:", filepath)

        results[prefix] = df
    
    return results
