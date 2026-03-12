library(data.table)
library(stringr)

data = fread("results/copilot_vs_registry_text_metrics2_new.csv")
data$pmcid=NULL 
data$registry_index=NULL 
data$publication_pmid=NULL 


mean_cal = function(file, metric){
  #function to calculate mean from all common metrics (in a row)

  #keep only specified metric 
  metric_values <- file[, .SD, .SDcols = patterns(metric)]
  metric_means <- rowMeans(metric_values, na.rm = TRUE)
  
  return(metric_means)
}


sd_cal = function(file, metric){
  #function to calculate standard deviation from all common metrics (in a row)
  
  #keep only specified metric 
  metric_values <- file[, .SD, .SDcols = patterns(metric)]
  metric_sds <- sd(metric_values, na.rm = TRUE)
  
  return(metric_sds)
}


# alphafold paper---------------------------------------------------------------
af = data[which(data$publication_title=="Highly accurate protein structure prediction with AlphaFold",)]


alphafold_stats <- data.frame(
  blue_mean = mean_cal(af, "bleu"),
  rougeL_mean = mean_cal(af, "rougeL"),
  meteor_mean = mean_cal(af, "meteor"),
  bertscore_mean = mean_cal(af, "bertscore"),
  blue_sd = sd_cal(af, "bleu"),
  rougeL_sd = sd_cal(af, "rougeL"),
  meteor_sd = sd_cal(af, "meteor"),
  bertscore_sd = sd_cal(af, "bertscore")

)


# write.table(alphafold_stats,
#             file = "alphafold_stats.txt",
#             sep = "\t",
#             row.names = FALSE,
#             quote     = FALSE)

# all papers---------------------------------------------------------------

col_mean = data[, lapply(.SD, mean, na.rm = TRUE), .SDcols = 2:ncol(data)]
col_sd = data[, lapply(.SD, sd, na.rm = TRUE), .SDcols = 2:ncol(data)]

all_stats <- data.frame(
  blue_mean = mean_cal(col_mean, "bleu"),
  rougeL_mean = mean_cal(col_mean, "rougeL"),
  meteor_mean = mean_cal(col_mean, "meteor"),
  bertscore_mean = mean_cal(col_mean, "bertscore"),
  blue_sd = mean_cal(col_sd, "bleu"),
  rougeL_sd = mean_cal(col_sd, "rougeL"),
  meteor_sd = mean_cal(col_sd, "meteor"),
  bertscore_sd = mean_cal(col_sd, "bertscore")
)



# write.table(all_stats,
#             file = "all_stats.txt",
#             sep = "\t",
#             row.names = FALSE,
#             quote     = FALSE)

# mean, min, max per column ----------------------------------------------------

stats_long <- rbindlist(
  lapply(names(data)[5:ncol(data)], function(col) {
    x <- data[[col]]
    data.table(
      variable = col,
      mean = mean(x, na.rm = TRUE),
      min  = min(x, na.rm = TRUE),
      min_row = which.min(x),
      max  = max(x, na.rm = TRUE),
      max_row = which.max(x)
    )
  })
)















