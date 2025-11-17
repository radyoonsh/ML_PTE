library(boot)
library(pROC)

# Define functions for sensitivity, specificity, accuracy, and F1 score
sensitivity <- function(actual, predicted) {
  TP <- sum(actual == 1 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  return(TP / (TP + FN))
}

specificity <- function(actual, predicted) {
  TN <- sum(actual == 0 & predicted == 0)
  FP <- sum(actual == 0 & predicted == 1)
  return(TN / (TN + FP))
}

accuracy <- function(actual, predicted) {
  correct <- sum(actual == predicted)
  return(correct / length(actual))
}

F1_score <- function(actual, predicted) {
  TP <- sum(actual == 1 & predicted == 1)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  return(2 * precision * recall / (precision + recall))
}

# Create a function to perform bootstrap resampling
bootstrap_metrics <- function(actual, predicted, metric_func, R = 1000) {
  boot_data <- data.frame(actual = actual, predicted = predicted)
  
  boot_fun <- function(data, indices) {
    metric_func(data$actual[indices], data$predicted[indices])
  }
  boot_results <- boot(boot_data, boot_fun, R = R)
  return(boot_results)
}


all_bootstrap_metrics <- function(actual, predicted, R = 1000) {
  sensitivity_bootstrap <- bootstrap_metrics(actual, predicted, sensitivity)
  specificity_bootstrap <- bootstrap_metrics(actual, predicted, specificity)
  accuracy_bootstrap <- bootstrap_metrics(actual, predicted, accuracy)
  F1_score_bootstrap <- bootstrap_metrics(actual, predicted, F1_score)
  
  sensitivity_ci <- quantile(sensitivity_bootstrap$t, c(0.025, 0.975))
  specificity_ci <- quantile(specificity_bootstrap$t, c(0.025, 0.975))
  accuracy_ci <- quantile(accuracy_bootstrap$t, c(0.025, 0.975))
  F1_score_ci <- quantile(F1_score_bootstrap$t, c(0.025, 0.975))
  
  metrics_results <- data.frame(specificity = specificity_ci,
                                sensitivity = sensitivity_ci,
                                accuracy = accuracy_ci,
                                F1 = F1_score_ci)
  return(metrics_results)
  
  
}

metrics_at <- function (truth, pred_positive, at = 0.95) {
sens_at <- at
spec_at <- at


proc <- roc(truth, pred_positive, direction = '<')
#To compute sensitivity and specificity at a threshold t,
#you must compare it with each of the observation o_i.
# With direction="<", o_i will be considered positive if o_i >= t, negative otherwise.
# With direction=">", o_i will be considered positive if o_i <= t, negative otherwise.

proc_results <- coords(proc, "all", ret = c("sensitivity", "specificity", "threshold"))
proc_results

# At the optimal point
proc_optimal <- coords(proc, "best") # optimal point
pred_class <- ifelse(pred_positive >= proc_optimal$threshold, 1, 0)
TP <- sum(truth == 1 & pred_class == 1)
TN <- sum(truth == 0 & pred_class == 0)
FP <- sum(truth == 0 & pred_class == 1)
FN <- sum(truth == 1 & pred_class == 0)
proc_optimal$accuracy <- sum (TP + TN) / sum(TP + TN + FP + FN)
proc_optimal$F1 <- 2 * ( (TP / (TP + FP)) *proc_optimal$sensitivity) / ((TP / (TP + FP)) + proc_optimal$sensitivity)

results <- bind_rows(proc_optimal, all_bootstrap_metrics(truth, pred_class))
cat('Metrics at the optimal point (Threshold: ', results[1,1], ')\n')
sens_ci <- paste0("Sensitivity: ", round(results[1, 3], 3), "(",
                  round(results[2, 3], 3), "–", round(results[3, 3], 3), ")")
cat(sens_ci, '\n')

spec_ci <- paste0("Specificity: ", round(results[1, 2], 3), "(",
                  round(results[2, 2], 3), "–", round(results[3, 2], 3), ")")
cat(spec_ci, '\n')

acc_ci <- paste0("Accuracy: ", round(results[1, 4], 3), "(",
                  round(results[2, 4], 3), "–", round(results[3, 4], 3), ")")
cat(acc_ci, '\n')

f1_ci <- paste0("F1: ", round(results[1, 5], 3), "(",
                 round(results[2, 5], 3), "–", round(results[3, 5], 3), ")")
cat(f1_ci, '\n\n')


# At the specific point of sensitivity
proc_sens <- proc_results %>%
  filter(sensitivity >= sens_at) %>%
  slice_min(sensitivity) %>%
  slice_max(specificity) %>%
  select(threshold, specificity, sensitivity)

pred_class <- ifelse(pred_positive >= proc_sens$threshold, 1, 0)
TP <- sum(truth == 1 & pred_class == 1)
TN <- sum(truth == 0 & pred_class == 0)
FP <- sum(truth == 0 & pred_class == 1)
FN <- sum(truth == 1 & pred_class == 0)

proc_sens$accuracy <- sum (TP + TN) / sum(TP + TN + FP + FN)
proc_sens$F1 <- 2 * ( (TP / (TP + FP)) *proc_sens$sensitivity) / ((TP / (TP + FP)) + proc_sens$sensitivity)

results <- bind_rows(proc_sens, all_bootstrap_metrics(truth, pred_class))

cat('Metrics at the Sensitivity ', sens_at, ' (Threshold: ', results[1,1], ')\n')
sens_ci <- paste0("Sensitivity: ", round(results[1, 3], 3), "(",
                  round(results[2, 3], 3), "–", round(results[3, 3], 3), ")")
cat(sens_ci, '\n')

spec_ci <- paste0("Specificity: ", round(results[1, 2], 3), "(",
                  round(results[2, 2], 3), "–", round(results[3, 2], 3), ")")
cat(spec_ci, '\n')

acc_ci <- paste0("Accuracy: ", round(results[1, 4], 3), "(",
                 round(results[2, 4], 3), "–", round(results[3, 4], 3), ")")
cat(acc_ci, '\n')

f1_ci <- paste0("F1: ", round(results[1, 5], 3), "(",
                round(results[2, 5], 3), "–", round(results[3, 5], 3), ")")
cat(f1_ci, '\n\n')


# At the specific point of specificity
proc_spec <- proc_results %>%
  filter(specificity >= spec_at) %>%
  slice_min(specificity) %>%
  slice_max(sensitivity) %>%
  select(threshold, specificity, sensitivity)

pred_class <- ifelse(pred_positive >= proc_spec$threshold, 1, 0)
TP <- sum(truth == 1 & pred_class == 1)
TN <- sum(truth == 0 & pred_class == 0)
FP <- sum(truth == 0 & pred_class == 1)
FN <- sum(truth == 1 & pred_class == 0)

proc_spec$accuracy <- sum (TP + TN) / sum(TP + TN + FP + FN)
proc_spec$F1 <- 2 * ( (TP / (TP + FP)) *proc_spec$sensitivity) / ((TP / (TP + FP)) + proc_spec$sensitivity)

results <- bind_rows(proc_spec, all_bootstrap_metrics(truth, pred_class))
cat('Metrics at the Specificity ' , spec_at, ' (Threshold: ', results[1,1], ')\n')
sens_ci <- paste0("Sensitivity: ", round(results[1, 3], 3), "(",
                  round(results[2, 3], 3), "–", round(results[3, 3], 3), ")")
cat(sens_ci, '\n')

spec_ci <- paste0("Specificity: ", round(results[1, 2], 3), "(",
                  round(results[2, 2], 3), "–", round(results[3, 2], 3), ")")
cat(spec_ci, '\n')

acc_ci <- paste0("Accuracy: ", round(results[1, 4], 3), "(",
                 round(results[2, 4], 3), "–", round(results[3, 4], 3), ")")
cat(acc_ci, '\n')

f1_ci <- paste0("F1: ", round(results[1, 5], 3), "(",
                round(results[2, 5], 3), "–", round(results[3, 5], 3), ")")
cat(f1_ci, '\n\n')


}
