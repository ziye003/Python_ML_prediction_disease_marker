# Databricks notebook source
library(data.table)
library(meta)
library(SparkR)


inverse_variance_fixed_effect_meta <- function(sum_dt, row_num, N) {
#   print(names(sum_dt))
  library(meta)
  row_num = unlist(row_num)
  estimate_list = unlist(sapply(1:N, function(n) paste("Estimate",as.character(n),sep="_")))
  se_list = unlist(sapply(1:N, function(n) paste("Std_Error",as.character(n),sep="_")))
  pv_list = unlist(sapply(1:N, function(n) paste("P",as.character(n),sep="_")))  
#   print(row_num)
  print(estimate_list)     
#   print(names(sum_dt))          
  estimate <- as.numeric(sum_dt[row_num, estimate_list, with = FALSE])  
  se <- as.numeric(sum_dt[row_num, se_list, with = FALSE])
  pv <- as.numeric(sum_dt[row_num, pv_list, with = FALSE])
#   print(dim(pv))
  res = metagen(estimate, se, sm="OR", method.tau = "HS") # OR
  frame = data.frame(
    meta_pvalue = res$pval.fixed,
    meta_or = exp(res$TE.fixed),
#     meta_or_lower_CI = exp(res$lower.fixed),
#     meta_or_upper_CI = exp(res$upper.fixed),
    meta_heterogeneity_pval = res$pval.Q,
    N_cohorts = sum(!is.na(estimate)),
    N_cohorts_valid = sum(pv <= 0.05, na.rm = T)
  )
  frame = data.frame(cbind(sum_dt[row_num, ], frame))
  return(frame)
}    


meta_analysis <- function(df, n_cohorts = 2){
  library(plyr)
  library(data.table)
  setDT(df)
  res = ldply(1:nrow(df), function(row_num) inverse_variance_fixed_effect_meta(df, row_num, n_cohorts), .parallel = TRUE)
  return (res)
} 


perform_meta_analysis <- function(sdf, n_cohorts){
    library(SparkR)
    sdf <- withColumn(sdf, "partition", round(rand() * 100))
    meta_dt <- gapplyCollect(
        sdf,
        c('partition'), 
        function(key, df){
            res = meta_analysis(df, n_cohorts)
            return(res)
        }
    )
  
    return(meta_dt[,!(names(meta_dt) %in% c('partition'))])
}
 




# COMMAND ----------


