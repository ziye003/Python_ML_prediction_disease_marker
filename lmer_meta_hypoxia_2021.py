# Databricks notebook source
# MAGIC %md #package & script

# COMMAND ----------

import os

import numpy as np

from pyspark.sql.functions import *

from pyspark.sql.functions import udf

import builtins

import pandas as pd

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages('meta')

# COMMAND ----------

# MAGIC %r
# MAGIC library(data.table)
# MAGIC library(meta)
# MAGIC library(SparkR)

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/alignment_util_oak

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/metaR_util_oak

# COMMAND ----------

# MAGIC %md # load alignment 

# COMMAND ----------

meta_path='/dbfs/mnt/client-002sap21p009-maltepe/04_data_analysis/alignment'
pos_alignment_grands=pd.read_csv('%salignment_grand_pos_4cohort_032522.csv'%meta_path)

# pos_alignment_grands=pd.read_csv('/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/raw_data/empathy_paint_pos_2cohort.csv')

pos_alignment_grands=pos_alignment_grands.iloc[:,1:]
pos_alignment_grands=pos_alignment_grands.astype(int, errors='ignore')
pos_alignment_grands=pos_alignment_grands.astype(str)
pos_alignment_grands_spark = spark.createDataFrame(pos_alignment_grands)
# pos_alignment_grands.sort_values('paint').head(3)

# COMMAND ----------

print(pos_alignment_grands.shape)
pos_alignment_grands.display()

# COMMAND ----------

meta_path='/dbfs/mnt/client-002sap21p009-maltepe/04_data_analysis/alignment'
neg_alignment_grands=pd.read_csv('%salignment_grand_neg_4cohort_032522.csv'%meta_path)


# neg_alignment_grands=pd.read_csv('/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/raw_data/empathy_paint_neg_2cohort.csv')
neg_alignment_grands=neg_alignment_grands.iloc[:,1:]
# neg_alignment_grands=neg_alignment_grands.astype(float)
neg_alignment_grands=neg_alignment_grands.astype(int, errors='ignore')
neg_alignment_grands=neg_alignment_grands.astype(str)
neg_alignment_grands_spark = spark.createDataFrame(neg_alignment_grands)
# neg_alignment_grands.sort_values('empathy').head(3)

# COMMAND ----------

# MAGIC %md # load regression

# COMMAND ----------

# path='/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/results/'

# p.to_csv('%sPAINT_std_imputed_hypoxia_spark.csv'%(path))


# p.to_csv('%sempathy_std_imputed_hypoxia_spark.csv'%(path))

# p.to_csv('%svalidation_std_imputed_hypoxia_spark.csv'%(path))
# p.to_csv('%smaltepe_std_imputed_hypoxia_spark.csv'%(path))

maltepe='/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/results/maltepe_std_imputed_hypoxia_spark.csv'
validation='/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/results/validation_std_imputed_hypoxia_spark.csv'
empathy='/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/results/empathy_std_imputed_hypoxia_spark.csv'
paint='/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/results/PAINT_std_imputed_hypoxia_spark.csv'


# COMMAND ----------

pt=pd.read_csv(maltepe)
pt.head()

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/alignment_util_oak

# COMMAND ----------

# regression files        
# reg_files = [maltepe,validation,empathy]
# cohorts = [ 'maltepe', 'validation','empathy']  
 

def get_spark_path(path):
    return path.replace('/dbfs', 'dbfs:')  


reg_files = [empathy,validation,maltepe,paint]
cohorts =[ 'empathy', 'validation','maltepe','paint']


# reg_files = [empathy,paint]
# cohorts =[ 'empathy', 'paint']

regression_df=pd.DataFrame(reg_files)
regression_df.index=cohorts

df_names=[]
for filename in reg_files:
    if filename.endswith('.csv'):
        
#         filename_list = filename.split('.')
#         temp = filename_list[0].split('_')
        temp=regression_df[regression_df[0]==filename].index.to_list()
        temp = ''.join(temp)
        df_name = '_'.join([temp, 'hypoxiaseverity', 'rLC'])
        df = spark.read.csv(get_spark_path(filename), header = True, inferSchema = True).drop('_c0')
        df.name = df_name
        df_names.append(df_name.lower())
        exec(f'{df_name.lower()} = df')

# COMMAND ----------

regression_files = []
reg_files = [empathy,validation,maltepe,paint]
# cohorts =[ 'empathy','paint']
# cohorts =[ 'maltepe', 'validation','empathy']  

# cohorts =[ 'empathy', 'validation','maltepe','paint']  

regressions =[
    n 
    for n in df_names
    for coh in cohorts
    if coh in n 
]  

# regressions = [
#     'cedars_covidseverity_rlc',
#  'chu_covidseveritypeak_rlc',
# #  'chu_covidseverity_rlc',
#  'merad_covidseveritypeak_rlc',
# #  'merad_covidseverity_rlc',
#  'umn_covidseveritypeak_rlc',
# #  'umn_covidseverity_rlc',
#  'wurfel_covidseverity_rlc']


[exec(f"regression_files.append({regression})") for regression in regressions] 

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/alignment_util_oak

# COMMAND ----------

meta_table_rlc_pos = generate_meta_table(regression_files, pos_alignment_grands_spark, mode = 'pos')

# COMMAND ----------

meta_table_rlc_pos.display()

# COMMAND ----------

meta_table_rlc_neg = generate_meta_table(regression_files, neg_alignment_grands_spark, mode = 'neg')

# COMMAND ----------

meta_table_rlc_neg.display()

# COMMAND ----------

meta_table_rlc_neg.createOrReplaceTempView('meta_table_rlc_neg')    
meta_table_rlc_pos.createOrReplaceTempView('meta_table_rlc_pos')  

# COMMAND ----------

meta_table_rlc_neg_df = meta_table_rlc_neg.toPandas()
meta_table_rlc_pos_df = meta_table_rlc_pos.toPandas()
print(meta_table_rlc_neg_df.shape)
print(meta_table_rlc_pos_df.shape)

# COMMAND ----------

meta_table_rlc_neg_df.display()

# COMMAND ----------

meta_table_rlc_df=pd.concat([meta_table_rlc_neg_df,meta_table_rlc_pos_df],axis=0)
meta_table_rlc_df.shape
meta_table_rlc_df=pd.DataFrame(meta_table_rlc_df)
meta_table_rlc_df.head()

# COMMAND ----------

meta_table_rlc_df.to_csv('/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/raw_data/pai_emp_mal_val_meta_table_032522.csv')

# COMMAND ----------

# MAGIC %md # meta analysis

# COMMAND ----------

# MAGIC %run /Users/Zi.Ye@sapient.bio/toolkits/metaR_util_oak

# COMMAND ----------

# MAGIC %r
# MAGIC meta_table_rlc_df=fread('/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/raw_data/pai_emp_mal_val_meta_table_032522.csv')

# COMMAND ----------

# MAGIC %r
# MAGIC display(meta_table_rlc_df)

# COMMAND ----------

# MAGIC %r
# MAGIC sparkR.session()
# MAGIC 
# MAGIC setDT(meta_table_rlc_df)
# MAGIC meta_table_bal <- createDataFrame(meta_table_rlc_df)
# MAGIC 
# MAGIC typeof(meta_table_bal)

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC library(stringr)
# MAGIC library(data.table)
# MAGIC # meta_table_bal = df
# MAGIC n_cohorts = str_sub( tail(names(meta_table_bal), 1), -1, -1)
# MAGIC n_cohorts 

# COMMAND ----------

# MAGIC %r
# MAGIC meta_table_bal
# MAGIC printSchema(meta_table_bal)
# MAGIC head(meta_table_bal)

# COMMAND ----------

# MAGIC %r
# MAGIC res = perform_meta_analysis(meta_table_bal, as.integer(n_cohorts))

# COMMAND ----------

# MAGIC %r
# MAGIC head(res)

# COMMAND ----------

# MAGIC %r
# MAGIC # fwrite(res, '/dbfs/mnt/client-002sap21p009-maltepe/04_data_analysis/meta_hypoxia_val_ori_paint.csv')
# MAGIC 
# MAGIC fwrite(res, '/dbfs/mnt/client-002sap21p024-paint/04_data_analysis/raw_data/meta_hypoxia_emp_paint_032522.csv')

# COMMAND ----------


