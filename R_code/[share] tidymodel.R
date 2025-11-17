rm(list=ls(all.names = TRUE))
set.seed(777)

# prepare parallel processing: 
cores <- parallel::detectCores(logical = TRUE)
# Create a cluster object and then register: 
cl <- parallel::makePSOCKcluster(cores)
doParallel::registerDoParallel(cl)

####################################################
# Data loading / cleaning
####################################################

library(tidyverse)
emr <- read_csv('../Data/share_raw_data.csv') %>% 
  mutate(age_thr = case_when(age <= 50 ~ 0.5, 
                             TRUE ~ age *0.01),
         ddimer.adjusted = ddimer - age_thr) %>%
  filter(is.na(dvt_equivocal) | dvt_equivocal != 1) %>%
  select(-age_thr) %>%
  select(-pte_seg, -dvt_equivocal)

emr <- emr[,1:125]


# Select Variables
vnames <- c("id", "sex", "age", "ddimer", "ddimer.adjusted", "pte", "bmi",
            "pr", "rr", "sbp", "dbp", "bt", "htn", "dm",
            "Albumin", "ALP", "ANC",  "aPTT", "BUN" , "Chol",
            "CK", "Cr", "CRP", "Eo", "Fbr",
            "Glu", "GOT", "GPT", "Hb", "Hct", "Lympho",
            "Phospho", "PLT", "Protein", "PT_p", "PT_i",
            "PT_s", "RBC", "Seg.neutro", "TnI", "Uric", "WBC",
            "prev_PTE_DVT", "surgery_bedrest", "htn_manual", "dm_manual",
            "ischemic_heart", "afib_af", "stroke", "active_cancer",
            "autoimmune", "coagulopathy", "ctx", "anticoagulation",
            "uni_leg_pain", "hemoptysis")

emr <- emr[vnames]


for (i in 43:56) {
  emr[[i]][is.na(emr[[i]])] <- 0
}


# Data cleansing: HTN, DM
emr$htn <- ifelse(emr$htn == 1 | emr$htn_manual == 1, 1, 0)
emr$dm <- ifelse(emr$dm == 1 | emr$dm_manual == 1, 1, 0)
emr <- emr[, -which(names(emr) %in% c("htn_manual", "dm_manual"))]


emr$pte <- ifelse(emr$pte == 'negative', 0, 1)
emr$sex <- ifelse(emr$sex == 'M', 0, 1)

vnames_new <- c("id", "Sex", "Age", "D_dimer", "Age_adjusted_D_dimer", "pte", "BMI",
            "PR", "RR", "SBP", "DBP", "BT", "HTN", "DM",
            "Albumin", "ALP", "ANC",  "aPTT", "BUN" , "Cholesterol",
            "CK", "Creatinine", "CRP", "Eosinophil", "Fibrinogen",
            "Glucose", "GOT", "GPT", "Hemoglobin", "Hematocrit", "Lymphocyte",
            "Phosphorus", "PLT", "Protein", "PT_percent", "PT_INR",
            "PT_second", "RBC", "Segmented_neutrophil", "Troponin_I", "Uric_acid", "WBC",
            "Prev_PTE_DVT", "Surgery_bedrest", 
            "Ischemic_heart_disease", "Afib_Af", "Stroke", "Active_cancer",
            "Autoimmune_disease", "Coagulopathy", "Chemotherapy", "Anticoagulation",
            "Unilateral_leg_pain", "Hemoptysis")

colnames(emr) <- vnames_new

emr <- emr %>%
  select(-PT_percent, -PT_second, -Age_adjusted_D_dimer) %>%
  select(-Creatinine, -BUN, -Uric_acid, -Glucose, -GOT, -GPT, -Phosphorus, -ALP)


####################################################
# kNN imputation
####################################################

library(VIM)

nona_ids <- emr %>%
  drop_na() %>%
  select(id)

vars_by_NAs <- emr %>%
  select(-id) %>%
  is.na() %>%
  colSums() %>%
  sort(decreasing = FALSE) %>%
  names()

one_imp <- emr %>%
  select(all_of(vars_by_NAs)) %>%
  kNN(k = 10, imp_var = FALSE) 

one_imp <- bind_cols(emr['id'], one_imp)


####################################################
# training, test dataset 
####################################################

test_ids <- sample(nona_ids$id, size = 500)

test_data <- one_imp %>%
  filter(id %in% test_ids) %>%
  select(-id) %>%
  mutate(pte = factor(pte, levels = c(1, 0)))

train_data <- one_imp %>%
  filter(!(id %in% test_ids))%>%
  select(-id) %>%
  mutate(pte = factor(pte, levels = c(1, 0)))


####################################################
# Tidymodels
####################################################

library(tidymodels)
library(vip)

pte_split <- make_splits(
  x = train_data,
  assessment = test_data
)

pte_training <- pte_split %>% training()
pte_test <- pte_split %>% testing()

pte_recipe <- recipe(pte ~.,
                     data = pte_training) %>%
  step_scale(all_predictors()) %>%
  step_normalize(all_predictors())

pte_metrics <- metric_set(roc_auc, sens, spec, accuracy) 

pte_folds <- vfold_cv(pte_training,
                      v = 5,
                      strata = pte)

baked_pte_training <- pte_recipe %>%
  prep(training = pte_training) %>%
  bake(new_data = NULL)



####################################################
# XGBoost
####################################################


library(xgboost)
set.seed(777)
xgb_model <- boost_tree(
  #min_n = tune(),           # Minimum number of observations in terminal nodes 
  #penalty_L1 = tune(),      #alpha L1 Regularization (Lasso)
  #penalty_L2 = tune(),      #lambda L2 Regularization (Ridge)
  trees = tune(),           # Number of boosting rounds 
  learn_rate = tune(),      # Learning rate 
  tree_depth = tune(),       # Maximum depth of trees 
  mtry = tune(),            # Number of predictors to randomly sample for each tree split 
  stop_iter = tune()  # Early stopping rounds for stopping criteria 
) %>%
  set_engine("xgboost", counts = FALSE) %>%  # Set the engine to XGBoost
  set_mode("classification")



xgb_tune_wkfl <- workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(pte_recipe)

xgb_grid <- expand.grid(
  trees = 220, 
  learn_rate = 0.02,  
  tree_depth = 8, 
  mtry = 0.1, 
  stop_iter = 50
)


xgb_tuning <- xgb_tune_wkfl %>%
  tune_grid(resamples = pte_folds,
            grid = xgb_grid,
            metrics = pte_metrics,
            control = control_grid(parallel_over = "resamples"))

autoplot(xgb_tuning)

xgb_tuning %>%
  collect_metrics()

xgb_tuning %>%  
  collect_metrics(summarize = FALSE) %>%  
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate), 
            median_roc_auc = median(.estimate), 
            max_roc_auc = max(.estimate)) 

xgb_tuning %>%  
  show_best(metric = 'roc_auc', n = 10) 

best_xgb_model <- xgb_tuning %>%  
  select_best(metric = 'roc_auc') 
best_xgb_model

final_xgb_wkfl <- xgb_tune_wkfl %>%  
  finalize_workflow(best_xgb_model)  
final_xgb_wkfl

xgb_final_fit <- final_xgb_wkfl %>%  
  last_fit(split = pte_split) 

xgb_final_fit %>%  
  collect_metrics() 

xgb_predictions <- xgb_final_fit %>% 
  collect_predictions()


# ROC curve
xgb_predictions %>%
  roc_curve(truth = pte, .pred_1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (XGBoost model)",
       x = "False Positive Rate",
       y = "True Positive Rate")

# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(xgb_predictions$pte), xgb_predictions$.pred_1, at = 0.9)

# confidence interval 
library(pROC)
ci.auc(xgb_predictions$pte, xgb_predictions$.pred_1, method = "delong")
ci.auc(xgb_predictions$pte, xgb_predictions$.pred_1, method = "bootstrap", boot.n = 1000)


# variable importance plot
library(DALEX)
library(DALEXtra)

pred <- function(model, newdata) {
  predicted <- predict(model, newdata, type = "prob")
  return(predicted$.pred_1)
}

explainer_xgb <- explain_tidymodels(
  extract_fit_parsnip(xgb_final_fit),
  data = baked_pte_training %>% select(-pte),
  y = 2- as.numeric(baked_pte_training$pte), 
  label = 'XGBoost',
  verbose = TRUE,
  predict_function = pred
)

set.seed(777)
vip_xgb <- model_parts(explainer_xgb, B = 30,
                       loss_function = loss_one_minus_auc)

plot_xgb_10 <- plot(vip_xgb, show_boxplots = FALSE, max_vars = 10,
                    title = "Variable Importance",
                    subtitle = "")

ggsave(filename = "./figure/xgb_vip_pte_only_10.tiff",
       plot = plot_xgb_10,
       width = 12, height = 13, dpi = 1500, units = "cm", compression = "lzw")


####################################################
# XGBoost threshold analysis
####################################################

new_data <- bind_cols(pte_test, xgb_predictions[".pred_1"])

ggplot(new_data %>% select(.pred_1, pte),
       aes(x = .pred_1, fill = factor(pte, levels = c(0, 1), labels = c("PTE negative", "PTE positive")) )) +
  geom_histogram(aes(y = after_stat(count) / tapply(after_stat(count), after_stat(PANEL), sum)[after_stat(PANEL)]),
                 position = "fill", binwidth = 0.1, color = "white") +
  scale_y_continuous(labels = scales::percent_format(), expand = c(0,0)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), labels = label_number(accuracy = 0.01), expand = c(0,0)) +
  scale_fill_manual(values = c("PTE negative" = "#90B3C9", "PTE positive" = "#E29494"), name = NULL) +
  labs(x = "Probability score", y = "Percentage") +
  theme(axis.line = element_line(color = "black"),
        panel.grid.major.y = element_line(color = "#444444", linetype = "dashed"),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size = 11),
        text = element_text(size = 14),
        panel.border = element_blank(),
        panel.background = element_blank(),
        panel.ontop = TRUE)


ggsave(filename = "./figure/ML_score_10.tiff",
       width = 24, height = 20, dpi = 1500, units = "cm", compression = "lzw")


#Screened patients using Different Thresholds

sens_table <- new_data %>%
  select(.pred_1, pte) %>%
  arrange(.pred_1) %>%
  mutate(patient = row_number()) %>%
  select(patient, .pred_1, pte) %>%
  mutate(pte = as.numeric(as.character(pte)),
         threshold = .pred_1,
         screened_number = patient,
         screened_number_percent = round(100 * screened_number / 500, 1),
         
         screened_pte = cumsum(pte),
         screened_pte_percent = round(100 * screened_pte / screened_number, 1),
         sensitivity = round(100 * (112 - screened_pte) / 112, 1),
         
         screened_normal = screened_number - screened_pte,
         screened_normal_percent = round(100 * screened_normal / screened_number, 1))%>%
  mutate(scr_total = paste0(screened_number, " (", screened_number_percent, "%)"),
         scr_normal = paste0(screened_normal, " (", screened_normal_percent, "%)"),
         scr_pte = paste0(screened_pte, " (", screened_pte_percent, "%)")) %>%
  select(sensitivity, threshold, scr_total, scr_normal, scr_pte)

sens_table2 <-  sens_table %>%
  group_by(sensitivity) %>%
  slice_max(threshold, n = 1) %>%
  ungroup() %>%
  arrange(desc(sensitivity))

find_closest_min <- function(ticks, df) {
  df %>%
    filter(sensitivity >= ticks) %>%
    slice_min(sensitivity, n = 1)
}

ticks <- seq(100, 0, by=-1)

sens_table3 <- bind_rows(lapply(ticks, find_closest_min, df = sens_table2)) %>%
  mutate(threshold = round(threshold, 3))
  
write_excel_csv(sens_table3, "../Data/sensitivity_threshold.csv")

####################################################
# Random Forest
####################################################


library(ranger)
set.seed(777)
rf_model <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n= tune()
) %>%
  set_engine("ranger") %>%
  set_mode("classification")


rf_tune_wkfl <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(pte_recipe)

rf_grid <- expand.grid(
  trees = 310, 
  mtry = 5, 
  min_n = 5 
)


rf_tuning <- rf_tune_wkfl %>%
  tune_grid(resamples = pte_folds,
            grid = rf_grid,
            metrics = pte_metrics,
            control = control_grid(parallel_over = "resamples"))

autoplot(rf_tuning)

rf_tuning %>%
  collect_metrics()

rf_tuning %>%  
  collect_metrics(summarize = FALSE) %>%  
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate), 
            median_roc_auc = median(.estimate), 
            max_roc_auc = max(.estimate)) 

rf_tuning %>%  
  show_best(metric = 'roc_auc', n = 10) 

best_rf_model <- rf_tuning %>%  
  select_best(metric = 'roc_auc') 
best_rf_model

final_rf_wkfl <- rf_tune_wkfl %>%  
  finalize_workflow(best_rf_model)  
final_rf_wkfl

rf_final_fit <- final_rf_wkfl %>%  
  last_fit(split = pte_split) 

rf_final_fit %>%  
  collect_metrics() 

rf_predictions <- rf_final_fit %>% 
  collect_predictions()

# ROC curve
rf_predictions %>%
  roc_curve(truth = pte, .pred_1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (Random Forest model)",
       x = "False Positive Rate",
       y = "True Positive Rate")



# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(rf_predictions$pte), rf_predictions$.pred_1, at = 0.9)

# confidence interval 
library(pROC)
ci.auc(rf_predictions$pte, rf_predictions$.pred_1, method = "delong")
ci.auc(rf_predictions$pte, rf_predictions$.pred_1, method = "bootstrap", boot.n = 1000)


# variable importance plot
library(DALEX)
library(DALEXtra)

pred <- function(model, newdata) {
  predicted <- predict(model, newdata, type = "prob")
  return(predicted$.pred_1)
}

explainer_rf <- explain_tidymodels(
  extract_fit_parsnip(rf_final_fit),
  data = baked_pte_training %>% select(-pte),
  y = 2- as.numeric(baked_pte_training$pte), 
  label = 'Random Forest',
  verbose = TRUE,
  predict_function = pred
)

set.seed(777)
vip_rf <- model_parts(explainer_rf, B = 30,
                       loss_function = loss_one_minus_auc)

plot_rf_10 <- plot(vip_rf, show_boxplots = FALSE, max_vars = 10,
                title = "Variable Importance",
                subtitle = "")

ggsave(filename = "./figure/rf_vip_pte_only_10.tiff",
       plot = plot_rf_10,
       width = 12, height = 13, dpi = 1500, units = "cm", compression = "lzw")


####################################################
# Logistic Regression
####################################################

library(stats)
set.seed(777)
select_significant_predictors <- function(dataframe) {
  significant_predictors <- c()
  
  # Iterate over each predictor
  for (column in names(dataframe)) {
    if (column != "pte") {  # Exclude the outcome variable
      # Construct formula dynamically
      formula <- as.formula(paste("pte ~", column))
      
      # Fit the univariate regression model
      model <- glm(formula, data = dataframe, family = "binomial")
      
      # Check p-value
      p_value <- summary(model)$coefficients[2, "Pr(>|z|)"]  # Get p-value for the predictor
      if (p_value < 0.05) {
        significant_predictors <- c(significant_predictors, column)
      }
    }
  }
  
  return(significant_predictors)
}

sig_predictors <- select_significant_predictors(baked_pte_training)

# backward elimination of variable
initial_formula <- as.formula(paste("pte~", paste(sig_predictors, collapse = ' + ')))
initial_model <- glm(initial_formula, data = baked_pte_training, family = "binomial")
selected_model <- stats::step(initial_model, direction = "backward")
var_names <- rownames(summary(selected_model)$coefficients)
selected_predictors <- var_names[2:length(var_names)]

# selected predictors -> tidymodel

pte_split_lr <- make_splits(
  x = train_data[c("pte", selected_predictors)],
  assessment = test_data[c("pte", selected_predictors)]
)

pte_training_lr <- pte_split_lr %>% training()

pte_recipe_lr <- recipe(pte ~ .,
                     data = pte_training_lr) %>%
  #step_corr(all_numeric(), threshold = 0.9) %>%
  step_scale(all_predictors()) %>%
  step_normalize(all_predictors())

lr_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

lr_wkfl <- workflow() %>%
  add_model(lr_model) %>%
  add_recipe(pte_recipe_lr)

lr_final_fit <- lr_wkfl %>%  
  last_fit(split = pte_split_lr) 

lr_final_fit %>%  
  collect_metrics() 

lr_predictions <- lr_final_fit %>% 
  collect_predictions()

# ROC curve
lr_predictions %>%
  roc_curve(truth = pte, .pred_1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (Logistic Regression model)",
       x = "False Positive Rate",
       y = "True Positive Rate")


# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(lr_predictions$pte), lr_predictions$.pred_1, at = 0.9)

# confidence interval 
library(pROC)
ci.auc(lr_predictions$pte, lr_predictions$.pred_1, method = "delong")
ci.auc(lr_predictions$pte, lr_predictions$.pred_1, method = "bootstrap", boot.n = 1000)


# variable importance plot
library(DALEX)
library(DALEXtra)

pred <- function(model, newdata) {
  predicted <- predict(model, newdata, type = "prob")
  return(predicted$.pred_1)
}

explainer_lr <- explain_tidymodels(
  extract_fit_parsnip(lr_final_fit),
  data = baked_pte_training %>% select(-pte),
  y = 2- as.numeric(baked_pte_training$pte), 
  label = 'Logistic Regression',
  verbose = TRUE,
  predict_function = pred
)

set.seed(777)
vip_lr <- model_parts(explainer_lr, B = 30,
                      loss_function = loss_one_minus_auc)

plot_lr_10 <- plot(vip_lr, show_boxplots = FALSE, max_vars = 10,
                title = "Variable Importance",
                subtitle = "")

ggsave(filename = "./figure/lr_vip_pte_only_10.tiff",
       plot = plot_lr_10,
       width = 12, height = 13, dpi = 1500, units = "cm", compression = "lzw")



####################################################
# GLMnet
####################################################

library(glmnet)
set.seed(777)
glmnet_model <- logistic_reg(
  penalty = tune(), # = lambda
  mixture = tune() # = alpha: 1: lasso, 0: ridge
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

glmnet_tune_wkfl <- workflow() %>%
  add_model(glmnet_model) %>%
  add_recipe(pte_recipe)

glmnet_grid <- expand.grid(
  penalty = 0.01, 
  mixture = 0.1 
)

glmnet_tuning <- glmnet_tune_wkfl %>%
  tune_grid(resamples = pte_folds,
            grid = glmnet_grid,
            metrics = pte_metrics,
            control = control_grid(parallel_over = "resamples"))

autoplot(glmnet_tuning)

glmnet_tuning %>%
  collect_metrics()

glmnet_tuning %>%  
  collect_metrics(summarize = FALSE) %>%  
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate), 
            median_roc_auc = median(.estimate), 
            max_roc_auc = max(.estimate)) 

glmnet_tuning %>%  
  show_best(metric = 'roc_auc', n = 5) 

best_glmnet_model <- glmnet_tuning %>%  
  select_best(metric = 'roc_auc') 
best_glmnet_model

final_glmnet_wkfl <- glmnet_tune_wkfl %>%  
  finalize_workflow(best_glmnet_model)  
final_glmnet_wkfl

glmnet_final_fit <- final_glmnet_wkfl %>%  
  last_fit(split = pte_split) 

glmnet_final_fit %>%  
  collect_metrics() 

glmnet_predictions <- glmnet_final_fit %>% 
  collect_predictions()

# ROC curve
glmnet_predictions %>%
  roc_curve(truth = pte, .pred_1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (GLMnet model)",
       x = "False Positive Rate",
       y = "True Positive Rate")



# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(glmnet_predictions$pte), glmnet_predictions$.pred_1, at = 0.9)

# confidence interval
library(pROC)
ci.auc(glmnet_predictions$pte, glmnet_predictions$.pred_1, method = "delong")
ci.auc(glmnet_predictions$pte, glmnet_predictions$.pred_1, method = "bootstrap", boot.n = 1000)


# variable importance plot
library(DALEX)
library(DALEXtra)

pred <- function(model, newdata) {
  predicted <- predict(model, newdata, type = "prob")
  return(predicted$.pred_1)
}

explainer_glmnet <- explain_tidymodels(
  extract_fit_parsnip(glmnet_final_fit),
  data = baked_pte_training %>% select(-pte),
  y = 2- as.numeric(baked_pte_training$pte), 
  label = 'Elastic net regression',
  verbose = TRUE,
  predict_function = pred
)

set.seed(777)
vip_glmnet <- model_parts(explainer_glmnet, B = 30,
                      loss_function = loss_one_minus_auc)

plot_glmnet_10 <- plot(vip_glmnet, show_boxplots = FALSE, max_vars = 10,
                    title = "Variable Importance",
                    subtitle = "")

ggsave(filename = "./figure/glmnet_vip_pte_only_10.tiff",
       plot = plot_glmnet_10,
       width = 12, height = 13, dpi = 1500, units = "cm", compression = "lzw")

plot_glmnet <- plot(vip_glmnet, show_boxplots = FALSE, max_vars = 5,
     title = "Variable Importance",
     subtitle = "")

plot_glmnet

ggsave(filename = "./figure/glmnet_vip_pte_only.tiff",
       width = 12, height = 8, dpi = 1500, units = "cm", compression = "lzw")


####################################################
# Support Vector Machine (linear)
####################################################

library(kernlab)
set.seed(777)
svm_linear_model <- svm_linear(
  cost = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_linear_tune_wkfl <- workflow() %>%
  add_model(svm_linear_model) %>%
  add_recipe(pte_recipe)

svm_linear_grid <- expand.grid(
  cost = 0.003 
)

svm_linear_tuning <- svm_linear_tune_wkfl %>%
  tune_grid(resamples = pte_folds,
            grid = svm_linear_grid,
            metrics = pte_metrics,
            control = control_grid(parallel_over = "resamples"))

autoplot(svm_linear_tuning)

svm_linear_tuning %>%
  collect_metrics()

svm_linear_tuning %>%  
  collect_metrics(summarize = FALSE) %>%  
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate), 
            median_roc_auc = median(.estimate), 
            max_roc_auc = max(.estimate)) 

svm_linear_tuning %>%  
  show_best(metric = 'roc_auc', n = 5) 

best_svm_linear_model <- svm_linear_tuning %>%  
  select_best(metric = 'roc_auc') 
best_svm_linear_model

final_svm_linear_wkfl <- svm_linear_tune_wkfl %>%  
  finalize_workflow(best_svm_linear_model)  
final_svm_linear_wkfl

svm_linear_final_fit <- final_svm_linear_wkfl %>%  
  last_fit(split = pte_split) 

svm_linear_final_fit %>%  
  collect_metrics() 

svm_linear_predictions <- svm_linear_final_fit %>% 
  collect_predictions()

# ROC curve
svm_linear_predictions %>%
  roc_curve(truth = pte, .pred_1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (Linear SVM model)",
       x = "False Positive Rate",
       y = "True Positive Rate")



# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(svm_linear_predictions$pte), svm_linear_predictions$.pred_1, at = 0.9)

# confidence interval 
library(pROC)
ci.auc(svm_linear_predictions$pte, svm_linear_predictions$.pred_1, method = "delong")
ci.auc(svm_linear_predictions$pte, svm_linear_predictions$.pred_1, method = "bootstrap", boot.n = 1000)


# variable importance plot
library(DALEX)
library(DALEXtra)

pred <- function(model, newdata) {
  predicted <- predict(model, newdata, type = "prob")
  return(predicted$.pred_1)
}

explainer_svm_linear <- explain_tidymodels(
  extract_fit_parsnip(svm_linear_final_fit),
  data = baked_pte_training %>% select(-pte),
  y = 2- as.numeric(baked_pte_training$pte), 
  label = 'Linear SVM',
  verbose = TRUE,
  predict_function = pred
)

set.seed(777)
vip_svm_linear <- model_parts(explainer_svm_linear, B = 30,
                      loss_function = loss_one_minus_auc)

plot_svm_linear_10 <- plot(vip_svm_linear, show_boxplots = FALSE, max_vars = 10,
                        title = "Variable Importance",
                        subtitle = "")

ggsave(filename = "./figure/svm_linear_vip_pte_only_10.tiff",
       plot = plot_svm_linear_10,
       width = 12, height = 13, dpi = 1500, units = "cm", compression = "lzw")


####################################################
# Support Vector Machine (radial)
####################################################

library(kernlab)
set.seed(777)
svm_radial_model <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_radial_tune_wkfl <- workflow() %>%
  add_model(svm_radial_model) %>%
  add_recipe(pte_recipe)

svm_radial_grid <- expand.grid(
  cost = 0.1, 
  rbf_sigma = 0.03 
)

svm_radial_tuning <- svm_radial_tune_wkfl %>%
  tune_grid(resamples = pte_folds,
            grid = svm_radial_grid,
            metrics = pte_metrics,
            control = control_grid(parallel_over = "resamples"))

autoplot(svm_radial_tuning)

svm_radial_tuning %>%
  collect_metrics()

svm_radial_tuning %>%  
  collect_metrics(summarize = FALSE) %>%  
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate), 
            median_roc_auc = median(.estimate), 
            max_roc_auc = max(.estimate)) 

svm_radial_tuning %>%  
  show_best(metric = 'roc_auc', n = 10) 

best_svm_radial_model <- svm_radial_tuning %>%  
  select_best(metric = 'roc_auc') 
best_svm_radial_model

final_svm_radial_wkfl <- svm_radial_tune_wkfl %>%  
  finalize_workflow(best_svm_radial_model)  
final_svm_radial_wkfl

svm_radial_final_fit <- final_svm_radial_wkfl %>%  
  last_fit(split = pte_split) 

svm_radial_final_fit %>%  
  collect_metrics() 

svm_radial_predictions <- svm_radial_final_fit %>% 
  collect_predictions()

# ROC curve
svm_radial_predictions %>%
  roc_curve(truth = pte, .pred_1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (Radial SVM model)",
       x = "False Positive Rate",
       y = "True Positive Rate")



# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(svm_radial_predictions$pte), svm_radial_predictions$.pred_1, at = 0.9)

# confidence interval 
library(pROC)
ci.auc(svm_radial_predictions$pte, svm_radial_predictions$.pred_1, method = "delong")
ci.auc(svm_radial_predictions$pte, svm_radial_predictions$.pred_1, method = "bootstrap", boot.n = 1000)


# variable importance plot
library(DALEX)
library(DALEXtra)

pred <- function(model, newdata) {
  predicted <- predict(model, newdata, type = "prob")
  return(predicted$.pred_1)
}

explainer_svm_radial <- explain_tidymodels(
  extract_fit_parsnip(svm_radial_final_fit),
  data = baked_pte_training %>% select(-pte),
  y = 2- as.numeric(baked_pte_training$pte), 
  label = 'Radial SVM',
  verbose = TRUE,
  predict_function = pred
)

set.seed(777)
vip_svm_radial <- model_parts(explainer_svm_radial, B = 30,
                              loss_function = loss_one_minus_auc)

plot_svm_radial_10 <- plot(vip_svm_radial, show_boxplots = FALSE, max_vars = 10,
                        title = "Variable Importance",
                        subtitle = "")

ggsave(filename = "./figure/svm_radial_vip_pte_only_10.tiff",
       plot = plot_svm_radial_10,
       width = 12, height = 13, dpi = 1500, units = "cm", compression = "lzw")

plot_svm_radial <- plot(vip_svm_radial, show_boxplots = FALSE, max_vars = 5,
     title = "Variable Importance",
     subtitle = "")

plot_svm_radial

ggsave(filename = "./figure/svm_radial_vip_pte_only.tiff",
       width = 12, height = 8, dpi = 1500, units = "cm", compression = "lzw")


####################################################
# Feedforward neural network (nnet)
####################################################

library(nnet)
set.seed(777)
nnet_model <- mlp(
  hidden_units = tune(),
  epochs = tune(),
  penalty = tune()
) %>% 
  set_engine("nnet", MaxNWts = 5000) %>%
  set_mode("classification")

nnet_tune_wkfl <- workflow() %>%
  add_model(nnet_model) %>%
  add_recipe(pte_recipe)

nnet_grid <- expand.grid(
  hidden_units = 5,
  epochs = 10, 
  penalty = 0.001 
)

nnet_tuning <- nnet_tune_wkfl %>%
  tune_grid(resamples = pte_folds,
            grid = nnet_grid,
            metrics = pte_metrics,
            control = control_grid(parallel_over = "resamples")) 

autoplot(nnet_tuning)

nnet_tuning %>%
  collect_metrics()

nnet_tuning %>%  
  collect_metrics(summarize = FALSE) %>%  
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate), 
            median_roc_auc = median(.estimate), 
            max_roc_auc = max(.estimate)) 

nnet_tuning %>%  
  show_best(metric = 'roc_auc', n = 5) 

best_nnet_model <- nnet_tuning %>%  
  select_best(metric = 'roc_auc') 
best_nnet_model

final_nnet_wkfl <- nnet_tune_wkfl %>%  
  finalize_workflow(best_nnet_model)  
final_nnet_wkfl

nnet_final_fit <- final_nnet_wkfl %>%  
  last_fit(split = pte_split) 

nnet_final_fit %>%  
  collect_metrics() 

nnet_predictions <- nnet_final_fit %>% 
  collect_predictions()

# ROC curve
nnet_predictions %>%
  roc_curve(truth = pte, .pred_1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (Neural network model)",
       x = "False Positive Rate",
       y = "True Positive Rate")

# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(nnet_predictions$pte), nnet_predictions$.pred_1, at = 0.9)

# confidence interval 
library(pROC)
ci.auc(nnet_predictions$pte, nnet_predictions$.pred_1, method = "delong")
ci.auc(nnet_predictions$pte, nnet_predictions$.pred_1, method = "bootstrap", boot.n = 1000)


# variable importance plot
library(DALEX)
library(DALEXtra)

pred <- function(model, newdata) {
  predicted <- predict(model, newdata, type = "prob")
  return(predicted$.pred_1)
}

explainer_nnet <- explain_tidymodels(
  extract_fit_parsnip(nnet_final_fit),
  data = baked_pte_training %>% select(-pte),
  y = 2- as.numeric(baked_pte_training$pte), 
  label = 'Neural network',
  verbose = TRUE,
  predict_function = pred
)

set.seed(777)
vip_nnet <- model_parts(explainer_nnet, B = 30,
                              loss_function = loss_one_minus_auc)

plot_nnet_10 <- plot(vip_nnet, show_boxplots = FALSE, max_vars = 10,
                  title = "Variable Importance",
                  subtitle = "")

ggsave(filename = "./figure/nnet_vip_pte_only_10.tiff",
       plot = plot_nnet_10,
       width = 12, height = 13, dpi = 1500, units = "cm", compression = "lzw")

plot_nnet <- plot(vip_nnet, show_boxplots = FALSE, max_vars = 5,
     title = "Variable Importance",
     subtitle = "")

plot_nnet

ggsave(filename = "./figure/nnet_vip_pte_only.tiff",
       width = 12, height = 8, dpi = 1500, units = "cm", compression = "lzw")

###################################
# Geneva Score
###################################

set.seed(777)
CC <- read_csv('../Data/after_review/leg_sx.csv')


geneva_test <- one_imp %>%
  filter(id %in% test_ids) %>%
  mutate(pte = factor(pte, levels = c(1, 0))) %>%
  left_join(CC, by = 'id') %>%
  mutate(gen_prev = ifelse(Prev_PTE_DVT == 1, 3, 0),
         gen_pr = case_when(PR >= 95 ~ 5,
                            PR >= 75 & PR <95 ~ 3,
                            TRUE ~ 0),
         gen_surgery = ifelse(Surgery_bedrest == 1, 2, 0),
         gen_hemoptysis = ifelse(Hemoptysis == 1, 2, 0),
         gen_cancer = ifelse(Active_cancer == 1, 2, 0),
         gen_unileg_pain = ifelse(Unilateral_leg_pain == 1, 3, 0),
         gen_unileg_edema = ifelse(Unilateral_leg_pain ==1 & Unilateral_leg_edema ==1, 4, 0),
         gen_age = ifelse(Age > 65, 1, 0),
         geneva = gen_prev + gen_pr + gen_surgery + gen_hemoptysis + gen_cancer +
           gen_unileg_pain + gen_unileg_edema + gen_age
  )

hist(geneva_test$geneva)

geneva_test %>%
  roc_curve(truth = pte, geneva) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = 'red') +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(title = "ROC Curve (Geneva score)",
       x = "False Positive Rate",
       y = "True Positive Rate")

geneva_test %>%
  roc_auc(truth = pte, geneva)

# metrics
source("./[share] metrics_at.R")
metrics_at(2 - as.numeric(geneva_test$pte), geneva_test$geneva, at = 0.9)

# confidence interval 
library(pROC)
ci.auc(geneva_test$pte, geneva_test$geneva, method = "bootstrap", boot.n = 1000)


###################################
# ROC curve combined
###################################

xgb_roc <- xgb_predictions %>%
  roc_curve(truth = pte, .pred_1)

rf_roc <- rf_predictions %>%
  roc_curve(truth = pte, .pred_1)

lr_roc <- lr_predictions %>%
  roc_curve(truth = pte, .pred_1)

glmnet_roc <- glmnet_predictions %>%
  roc_curve(truth = pte, .pred_1)

svm_linear_roc <- svm_linear_predictions %>%
  roc_curve(truth = pte, .pred_1)

svm_radial_roc <- svm_radial_predictions %>%
  roc_curve(truth = pte, .pred_1)

nnet_roc <- nnet_predictions %>%
  roc_curve(truth = pte, .pred_1)

geneva_roc <- geneva_test %>%
  roc_curve(truth = pte, geneva)

xgb_roc %>%
  mutate(Model = 'XGBoost') %>%
  bind_rows(., rf_roc %>% mutate(Model = 'Random forest'),
            lr_roc %>% mutate(Model = 'Logistic regression'),
            glmnet_roc %>% mutate(Model = 'Elastic net regression'),
            svm_linear_roc %>% mutate(Model = 'SVM (Linear kernel)'),
            svm_radial_roc %>% mutate(Model = 'SVM (Radial kernel)'),
            nnet_roc %>% mutate(Model = 'Neural network'),
            geneva_roc %>% mutate(Model = 'Geneva score')) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_path(linewidth = 0.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = 'gray') +  # Add diagonal line for reference
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(x = "False Positive Rate",
       y = "True Positive Rate")+
  scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#882E72"),
                     limits = c('XGBoost', 'Random forest', 'Logistic regression', 'Elastic net regression', 'SVM (Linear kernel)', 'SVM (Radial kernel)', 'Neural network', 'Geneva score'))

ggsave(filename = "./figure/ROC_all_models_pte_only.tiff",
       width = 18, height = 18, dpi = 1500, units = "cm", compression = "lzw")



###################################
# VIP combined
###################################

library(gridExtra)

# Arrange the plots in a 3:2 grid
combined_plot_10 <- grid.arrange(plot_xgb_10, plot_rf_10, plot_lr_10, plot_glmnet_10,
                                 plot_svm_linear_10, plot_svm_radial_10,
                                 plot_nnet_10, nrow=4, ncol=2)


ggsave(filename = "./figure/all_vip_pte_only_10.tiff",
       plot = combined_plot_10,
       width = 24, height = 50, dpi = 1500, units = "cm", compression = "lzw")

ggsave(filename = "./figure/all_vip_pte_only_10.pdf", device = "pdf",
       plot = combined_plot_10,
       width = 24, height = 50, dpi = 1500, units = "cm")



