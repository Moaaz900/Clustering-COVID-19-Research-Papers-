# Databricks notebook source
# DBTITLE 1,CORD-19 Clustering Project
"""

===============================================================
Prepared By:

    1. Ahmed El-Sayed Hamdan - 191035
    2. Mohamed Ahmed Elkhateeb - 191017
    3. Moaaz Youssef Ghonaimy - 191036
    4. Maryam Akram Elghalban - 191084
===============================================================

Documenation for the work done:
------------------------------------------------
1.	Read the dataset using spark
    a.	 using databricks.com by creating a cluster and loading the data directly
    b.	Check the number of partitions for each part of data
    c.	Repartition the data into 2 partitions because the cluster has one node 2 cores
    d.	Store the data as parquet
    e.	Reload the data again but from parquet files which save a lot of time
    i.	JSON files take about 16.87 minutes
    ii.	Parquet files take about 4.76 seconds

2.	Exploratory data analysis (EDA)
    a.	Print Schema to check the available columns 
    b.	Get most important columns (paper_id, title, abstract, body_text, back_matter)
    c.	Add “source” column and set the value with “comm, uncomm, bio”
    d.	Check about null and empty for all of these columns and this is the result
        •	null titles: 0 
        •	empty titles: 617
        •	null abstracts: 0
        •	abstracts has less than 100 char: 1624 
        •	abstracts has less than 100 char and not empty: 107
        •	null body_text: 0
        •	body_text has less than 10000 char: 1188
        •	body_text has less than 10000 char and not empty: 1188
        •	null back_matter: 0
        •	back_matter has less than 100 char: 3776
        •	back_matter has less than 100 char and not empty: 974
    e.	check about duplicate title and there are about 500 duplicate titles
    f.	check about language and there are about 100 documents non English (we filter step by step so these 5 after removing null duplicates and short body_text) using langdetect library

3.	Preparation and Cleaning the data
    a.	drop null and empty and short body text records
    b.	drop duplicate titles
    c.	keep only English document

4.	Preprocessing
    a.	Remove punctuations
    b.	Remove stop words
    c.	Remove custom stop words
    d.	convert text to lower case

5.	Vectorization 
    We use TF-IDF. 
    Here we try without using the feature_num parameters then it produce about 250k features, but unfortunately when apply PCA, it limited to only 16K features, so we limit the features num to 16K but then PCA has out of memery exception so then we decrease the features_num again and again with a lot of values (50000,10000,5000,1000,500)

6.	Clustering
    a.	Use Kmeans
    b.	Use PCA to reduce dimensions 

7.	Recommender system
    Unfortunately we didn’t start on this part

"""

# COMMAND ----------

# DBTITLE 1,Loading the data from json files
# Configure Paths for the json files in databricks
comm_use_subset_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/comm_use_subset/comm_use_subset/"
noncomm_use_subset_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/noncomm_use_subset/noncomm_use_subset/"
biorxiv_medrxiv_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/"
json_schema_path = "/databricks-datasets/COVID/CORD-19/2020-03-13/json_schema.txt"

# COMMAND ----------

# start loading the files
comm_use_subset = spark.read.option("multiLine", True).json(comm_use_subset_path)
noncomm_use_subset = spark.read.option("multiLine", True).json(noncomm_use_subset_path)
biorxiv_medrxiv = spark.read.option("multiLine", True).json(biorxiv_medrxiv_path)
json_schema_path = spark.read.option("multiLine", True).json(json_schema_path)

# COMMAND ----------

# DBTITLE 1,Check current partition counts
# Get number of partitions for each group of files
comm_use_subset.rdd.getNumPartitions()
noncomm_use_subset.rdd.getNumPartitions()
biorxiv_medrxiv.rdd.getNumPartitions()

# COMMAND ----------

# DBTITLE 1,Configure Parquet Paths in Python
#Configure Parquet Paths to be used to store the files and load later
comm_use_subset_pq_path = "/tmp/parquet/comm_use_subset.parquet"
noncomm_use_subset_pq_path = "/tmp/parquet/noncomm_use_subset.parquet"
biorxiv_medrxiv_pq_path = "/tmp/parquet/biorxiv_medrxiv/biorxiv_medrxiv.parquet"

# COMMAND ----------

# DBTITLE 1,Write out in Pqarquet format in 2 partitions
# Write out in Pqarquet format in 2 partitions
# Note, this cluster has 2 nodes
comm_use_subset.repartition(2).write.format("parquet").mode("overwrite").save(comm_use_subset_pq_path)
noncomm_use_subset.repartition(2).write.format("parquet").mode("overwrite").save(noncomm_use_subset_pq_path)
biorxiv_medrxiv.repartition(2).write.format("parquet").mode("overwrite").save(biorxiv_medrxiv_pq_path)

# COMMAND ----------

# DBTITLE 1,Reread files form stored Parquet 
comm_use_subset = spark.read.format("parquet").load(comm_use_subset_pq_path)
noncomm_use_subset = spark.read.format("parquet").load(noncomm_use_subset_pq_path)
biorxiv_medrxiv = spark.read.format("parquet").load(biorxiv_medrxiv_pq_path)

# COMMAND ----------

# DBTITLE 1,Print schema for 3 subset to explore the structure and how we can get the information
comm_use_subset.printSchema()

# COMMAND ----------

noncomm_use_subset.printSchema()

# COMMAND ----------

biorxiv_medrxiv.printSchema()

# COMMAND ----------

# DBTITLE 1,Show data
display(comm_use_subset)

# COMMAND ----------

display(comm_use_subset.select('abstract.text'))

# COMMAND ----------

noncomm_use_subset.show(5)

# COMMAND ----------

biorxiv_medrxiv.show(5)

# COMMAND ----------

# DBTITLE 1,Select the needed columns only from the dataset - comm_use_subset
from pyspark.sql.functions import col, lit, concat_ws, explode

# select columns which have text like title, abstract, body and backmatter
# select author object to make some processing later
# select paperid
comm_use_subset_selected = comm_use_subset.select(
            col("metadata.title").alias("title"),
            col("abstract.text").alias("abstract_arr"),
            col("body_text.text").alias("body_text_arr"),
            col("back_matter.text").alias("back_matter_arr"),
          #  col("metadata.authors").alias("authors_arr"),
            "paper_id")

# concatenate the paragraphs in one text for abstartc, body_text and back_matter
comm_use_subset_selected = comm_use_subset_selected.withColumn('abstract', concat_ws(' ', 'abstract_arr'))
comm_use_subset_selected = comm_use_subset_selected.withColumn('body_text', concat_ws(' ', 'body_text_arr'))
comm_use_subset_selected = comm_use_subset_selected.withColumn('back_matter', concat_ws(' ', 'back_matter_arr'))

# add column for source dataset
comm_use_subset_selected = comm_use_subset_selected.withColumn("source",lit("comm_use_subset"))

# drop unneeded columns 
comm_use_subset_selected = comm_use_subset_selected.drop("abstract_arr","body_text_arr","back_matter_arr")

comm_use_subset_selected.printSchema()

# COMMAND ----------

# DBTITLE 1,Select the needed columns only from the dataset - noncomm_use_subset
# select columns which have text like title, abstract, body and backmatter
# select author object to make some processing later
# select paperid
noncomm_use_subset_selected = noncomm_use_subset.select(
            col("metadata.title").alias("title"),
            col("abstract.text").alias("abstract_arr"),
            col("body_text.text").alias("body_text_arr"),
            col("back_matter.text").alias("back_matter_arr"),
            #col("metadata.authors").alias("authors_arr"),
            "paper_id")

# concatenate the paragraphs in one text for abstartc, body_text and back_matter
noncomm_use_subset_selected = noncomm_use_subset_selected.withColumn('abstract', concat_ws(', ', 'abstract_arr'))
noncomm_use_subset_selected = noncomm_use_subset_selected.withColumn('body_text', concat_ws(', ', 'body_text_arr'))
noncomm_use_subset_selected = noncomm_use_subset_selected.withColumn('back_matter', concat_ws(', ', 'back_matter_arr'))

# add column for source dataset
noncomm_use_subset_selected = noncomm_use_subset_selected.withColumn("source",lit("noncomm_use_subset"))

# drop unneeded columns 
noncomm_use_subset_selected = noncomm_use_subset_selected.drop("abstract_arr","body_text_arr","back_matter_arr")

noncomm_use_subset_selected.printSchema()

# COMMAND ----------

# DBTITLE 1,Select the needed columns only from the dataset - biorxiv_medrxiv
# select columns which have text like title, abstract, body and backmatter
# select author object to make some processing later
# select paperid
biorxiv_medrxiv_selected = biorxiv_medrxiv.select(
            col("metadata.title").alias("title"),
            col("abstract.text").alias("abstract_arr"),
            col("body_text.text").alias("body_text_arr"),
            col("back_matter.text").alias("back_matter_arr"),
             # col("metadata.authors").alias("authors_arr"),
            "paper_id")

# concatenate the paragraphs in one text for abstartc, body_text and back_matter
biorxiv_medrxiv_selected = biorxiv_medrxiv_selected.withColumn('abstract', concat_ws(', ', 'abstract_arr'))
biorxiv_medrxiv_selected = biorxiv_medrxiv_selected.withColumn('body_text', concat_ws(', ', 'body_text_arr'))
biorxiv_medrxiv_selected = biorxiv_medrxiv_selected.withColumn('back_matter', concat_ws(', ', 'back_matter_arr'))

# add column for source dataset
biorxiv_medrxiv_selected = biorxiv_medrxiv_selected.withColumn("source",lit("biorxiv_medrxiv"))

# drop unneeded columns 
biorxiv_medrxiv_selected = biorxiv_medrxiv_selected.drop("abstract_arr","body_text_arr","back_matter_arr")

biorxiv_medrxiv_selected.printSchema()


# COMMAND ----------

# DBTITLE 1,Concatenate all datasets in one dataframe
# Concatenate all datasets in one dataframe (comm_use_subset_selected, noncomm_use_subset_selected, biorxiv_medrxiv_selected)
# after selecting the neeeded columsn only
all_data_df = comm_use_subset_selected.union(noncomm_use_subset_selected).union(biorxiv_medrxiv_selected)
all_data_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Count the data
# get the counts for each group of records
from pyspark.sql.functions import count, countDistinct
display(all_data_df.groupBy("source").agg(count("paper_id")))

# COMMAND ----------

# DBTITLE 1,Check null or empty title
from pyspark.sql.functions import length
from pyspark.sql.functions import regexp_replace, col

# check number of null titles
print("Number of null titles:  ", all_data_df.select("title").where(col("title").isNull()).count())

# check the values of title which has columns empty
count_titles = all_data_df.select("title").where(col("title") == "").count()
print("Number of empty titles: ", count_titles)

# COMMAND ----------

# DBTITLE 1,Check null or empty Abstract 
# check number of null abstratcs
print("Number of null abstracts:  ", all_data_df.select("abstract").where(col("abstract").isNull()).count())

# check the values of abstract which has columns less than 100 charcters which is so small
count_abstratcs = all_data_df.select("abstract").where(length(col("abstract")) < 100 ).count()
print("Number of abstracts has less than 100 char: ", count_abstratcs)

# check the values of abstract which has columns less than 100 charcters and not empty
count_abstratcs2 = all_data_df.select("abstract").where(length(col("abstract")) < 100 ).where(col("abstract") != "" ).count()
print("Number of abstracts has less than 100 char and not empty: ", count_abstratcs2)

# COMMAND ----------

#display the abstratcs which is not empty and has less than 100 char
display(all_data_df.select("abstract").where(length(col("abstract")) < 100 ).where(col("abstract") != "" ))

# COMMAND ----------

# DBTITLE 1,Check null or empty body_text
# check number of null body_text
print("Number of null body_text:  ", all_data_df.select("body_text").where(col("body_text").isNull()).count())

# check the values of body_text which has columns less than 100 charcters which is so small
count_body_text = all_data_df.select("body_text").where(length(col("body_text")) < 10000 ).count()
print("Number of body_text has less than 10000 char: ", count_body_text)

# check the values of body_text which has columns less than 10000 charcters and not empty
count_body_text2 = all_data_df.select("body_text").where(length(col("body_text")) < 10000 ).where(col("body_text") != "" ).count()
print("Number of body_text has less than 10000 char and not empty: ", count_body_text2)

# COMMAND ----------

#display the body_text which is not empty and has less than 100 char
display(all_data_df.select("body_text").where(length(col("body_text")) < 10000 ).where(col("body_text") != "" ))

# COMMAND ----------

#display the body_text which is not empty and has less than 1000 char for one record
print(all_data_df.select("body_text").where(length(col("body_text")) < 10000 ).where(col("body_text") != "" ).collect()[0][0])

# COMMAND ----------

# DBTITLE 1,Check null or empty back_matter
# check number of null back_matter
print("Number of null back_matter:  ", all_data_df.select("back_matter").where(col("back_matter").isNull()).count())

# check the values of back_matter which has columns less than 100 charcters which is so small
count_back_matter = all_data_df.select("back_matter").where(length(col("back_matter")) < 100 ).count()
print("Number of back_matter has less than 100 char: ", count_back_matter)

# check the values of back_matter which has columns less than 100 charcters and not empty
count_back_matter2 = all_data_df.select("back_matter").where(length(col("back_matter")) < 100 ).where(col("back_matter") != "" ).count()
print("Number of back_matter has less than 100 char and not empty: ", count_back_matter2)


# COMMAND ----------

#display the back_matter which is not empty and has less than 100 char
display(all_data_df.select("back_matter").where(length(col("back_matter")) < 100 ).where(col("back_matter") != "" ))

# COMMAND ----------

"""After exploring the data we found 3 options:
      1. Use title to be features
      2. Use abstract to be features
      3. Use body_text to be features
      4. merge abstract and body_text
      
   So we try all of them and unfortunaty there is no big difference between options in the clustreing results
"""

# COMMAND ----------

# DBTITLE 1,Merge abstract and body_text together
from pyspark.sql.functions import concat, coalesce  
all_data_df = all_data_df.withColumn("all_text", concat(coalesce(col("abstract")), coalesce(col("body_text"))))

# COMMAND ----------

# DBTITLE 1,Remove empty title or missing text records
from pyspark.sql.functions import length
all_data_df = all_data_df.where(col("title") != "" ).where(length(col("title")) > 50 )

# COMMAND ----------

# DBTITLE 1,Remove duplicate records
print("total count of records: ", all_data_df.count())
print("total count of unique titles: ", all_data_df.select("title").distinct().count())

#remove duplicate title records 
all_data_df = all_data_df.dropDuplicates(["title"])
print("total count of records after remove duplicate: ", all_data_df.count())

# COMMAND ----------

# DBTITLE 1,Detect the language of the body_text
from langdetect import detect
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

def language_detect(data_str):
    lang = detect(data_str) 
    return lang

language_detect_udf = udf(language_detect, StringType())
all_data_df = all_data_df.withColumn('language', language_detect_udf(all_data_df.title))
all_data_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Remove none English records
print("total count of records: ", all_data_df.count())
print("total count of non English records: ", all_data_df.where(col("language") != 'en').count())

#remove non English records 
all_data_df = all_data_df.where(col("language") == 'en')
print("total count of English records only: ", all_data_df.count())

# COMMAND ----------

# DBTITLE 1,tokenize the text and make it lower case and remove punctuation 
from pyspark.ml.feature import RegexTokenizer

regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'title', outputCol = 'text_token')

all_data_df_token = regexTokenizer.transform(all_data_df)
all_data_df_token.select("paper_id", "title", "text_token").show(3)

# COMMAND ----------

display(all_data_df_token.select("title", "text_token"))

# COMMAND ----------

# DBTITLE 1,Remove Stop Words
from pyspark.ml.feature import StopWordsRemover
swr = StopWordsRemover(inputCol = 'text_token', outputCol = 'text_sw_removed')
print(swr.getStopWords())
all_data_df_sw = swr.transform(all_data_df_token)

# COMMAND ----------

# DBTITLE 1,Remove custom stop words
custom_stop_words = ['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org',
'https', 'et', 'al', 'author', 'figure','rights', 'reserved', 'permission', 'used', 'using',
'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI', 'www']

cswr = StopWordsRemover(inputCol = 'text_sw_removed', outputCol = 'text_csp_removed')
cswr.setStopWords(custom_stop_words)
print(cswr.getStopWords())
all_data_df_sw = cswr.transform(all_data_df_sw)
all_data_df_sw.select("paper_id", "title", "text_token","text_sw_removed","text_csp_removed").show(3)

# COMMAND ----------

display(all_data_df_sw.select("title", "text_token","text_csp_removed"))

# COMMAND ----------

# DBTITLE 1,Vectorization 
#TF-IDF. This will convert our string formatted data into a measure of how important
#each word is to the instance out of the record as a whole.

from pyspark.ml.feature import HashingTF
#65535
hashingTF = HashingTF(inputCol="text_csp_removed", outputCol="rawFeatures", numFeatures=250)
all_data_df_rawFeatures = hashingTF.transform(all_data_df_sw)
#all_data_df_rawFeatures.select("paper_id","text_csp_removed", "rawFeatures").show(3)

# COMMAND ----------

from pyspark.ml.feature import IDF
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
all_data_df_features = idf.fit(all_data_df_rawFeatures).transform(all_data_df_rawFeatures)
#all_data_df_features.select("paper_id","text_csp_removed", "rawFeatures", "features").show(5)

# COMMAND ----------

# DBTITLE 1,Apply PCA
from pyspark.ml.feature import  PCA
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
all_data_df_features_pca = pca.fit(all_data_df_features)

# COMMAND ----------

all_data_df_features_pca = all_data_df_features_pca.transform(all_data_df_features)
all_data_df_features_pca.select("paper_id", "features", "pcaFeatures").show(3)

# COMMAND ----------

# DBTITLE 1,Elbow Method
# ----- inprogress ------
import numpy as np
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans()\
            .setK(k)\
            .setSeed(1) \
            .setFeaturesCol("pcaFeatures")\
            .setPredictionCol("cluster")

    model = kmeans.fit(all_data_df_features_pca)
    cost[k] = model.computeCost(all_data_df_features_pca)
    print("k: " , k," ", cost[k])

# COMMAND ----------

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# COMMAND ----------

# DBTITLE 1,Clustering using KMeans
from pyspark.ml.clustering import KMeans

# Loads data.
dataset = all_data_df_features_pca.select("pcaFeatures")

# Trains a k-means model.
kmeans = KMeans().setK(3).setFeaturesCol("pcaFeatures")
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator().setFeaturesCol("pcaFeatures")

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

display(predictions.groupBy("prediction").count()) 

# COMMAND ----------

# DBTITLE 1,Using pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[regexTokenizer, swr,cswr, hashingTF, idf,pca, kmeans])

model = pipeline.fit(all_data_df)
results = model.transform(all_data_df)

display(results.groupBy("prediction").count()) 
