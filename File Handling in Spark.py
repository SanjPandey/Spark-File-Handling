!sudo apt update
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
#Check this site for the latest download link https://www.apache.org/dyn/closer.lua/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
!wget -q https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
!tar xf spark-3.2.1-bin-hadoop3.2.tgz
!pip install -q findspark
!pip install pyspark
!pip install py4j

import os
import sys
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"


import findspark
findspark.init()
findspark.find()

import pyspark

from pyspark.sql import DataFrame, SparkSession
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F

spark= SparkSession \
       .builder \
       .appName("Our First Spark Example") \
       .getOrCreate()

spark

xxxxxxxxxxxxxxxxxx

from google.colab import output
output.serve_kernel_port_as_window(4040, path='/jobs/index.html')


#File Handling In Spark

from google.colab import files # upload file from local System
uploaded = files.upload()

from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("CSV to Parquet Example") \
    .getOrCreate()

# Read CSV file into DataFrame
df = spark.read.csv("iris.csv", header=True, inferSchema=True)
parquet_output_path = "/content/outputrecord"

df.write.mode("overwrite").parquet(parquet_output_path)

df1 = spark.read.parquet('outputrecord')
df1.show()

#DF1 = SPARK.READ.PARQUEST('OUTPUT')
##parquet_output1 = "/content/output1"

# Save DataFrame as Parquet
#df.write.mode("overwrite").parquet(parquet_output1)

#df1=spark.read.parquet('parquet_output1')

# Show the DataFrame schema and some sample data
print("CSV DataFrame Schema:")
df.printSchema()
print("Sample data from CSV:")
df.show()

xxxxxxxxxxxxxxxxxx
Sql Query in Spark

To Run Sql Query we use createOrReplaceTempView('test') syntax
now first import package
from pyspark.sql import SparkSession - this package is used to
 access sql query
 
 from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("CSV to Parquet Example") \
    .getOrCreate()

# Read CSV file into DataFrame
df = spark.read.csv("bank_note_data.csv", header=True, inferSchema=True)
print("CSV DataFrame Schema:")
df.printSchema()
print("Sample data from CSV:")
df.show()
 
 df.createOrReplaceTempView('test1')
 #Execute SQL query using the temporary view
result = spark.sql("SELECT * FROM test1 WHERE class=0 ")
 result.show()