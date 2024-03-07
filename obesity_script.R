#my first Kaggle Competition
rm(list=ls())
pacman::p_load("tidyverse", #for tidy data science practice
               "tidymodels", "workflows", "finetune", "themis", "embed", "butcher",# for tidy machine learning
               "pacman", #package manager
               "devtools", #developer tools
               "Hmisc", "skimr", "broom", "modelr",#for EDA
               "jtools", "huxtable", "interactions", # for EDA
               "ggthemes", "ggstatsplot", "GGally",
               "scales", "gridExtra", "patchwork", "ggalt", "vip",
               "ggstance", "ggfortify", # for ggplot
               "DT", "plotly", #interactive Data Viz
               # Lets install some ML related packages that will help tidymodels::
               "usemodels", "poissonreg", "agua", "sparklyr", "dials", "bonsai",#load computational engines
               "doParallel", # for parallel processing (speedy computation)
               "ranger", "xgboost", "glmnet", "kknn", "earth", "klaR", "discrim", "naivebayes", "baguette", "kernlab", "lightgbm",#random forest
               "janitor", "lubridate")


# import split data
set.seed(2024030601)
data_train <- read_csv("train.csv")

# check missing data
table(is.na(data_train))

# check imbalance
data_train %>% count(NObeyesdad) %>% mutate(prop = n/sum(n))

# get variables into correct class
data_train <-
  data_train %>% 
  mutate_if(is.character, as.factor)

# check correlation
data_train %>%
  select_if(is.numeric) %>%
  as.matrix(.) %>%
  rcorr() %>%
  tidy() %>%
  mutate(absCorr = abs(estimate)) %>%
  arrange(desc(absCorr)) %>% 
  dplyr::select(-estimate, -n, - p.value) %>% 
  DT::datatable() %>% 
  formatRound("absCorr", digits = 3)

# create data_fold
data_fold <-
  data_train %>% 
  vfold_cv(v = 10, strata = NObeyesdad)

# create recipes
base_rec <-
  recipes::recipe(formula = NObeyesdad ~.,
                  data = data_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_nzv(all_predictors()) %>% # remove zero variance
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

age_rec <-
  recipes::recipe(formula = NObeyesdad ~.,
                  data = data_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_nzv(all_predictors()) %>% # remove zero variance
  step_mutate(age_group = Age) %>% 
  step_cut(age_group,
           breaks = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
           include_outside_range = TRUE) %>%
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

#build models

# random forest
rf_spec <-
  rand_forest() %>% 
  set_engine("ranger",
             importance = "impurity") %>% 
  set_mode("classification") %>% 
  set_args(trees = tune(),
           mtry = tune(),
           min_n = tune())

# xgboost
xgb_spec <-
  boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>% 
  set_args(trees = tune(),
           tree_depth = tune(),
           min_n = tune(),
           loss_reduction = tune(),
           sample_size = tune(),
           mtry = tune(),
           learn_rate = tune(),
           stop_iter = 10)

# knn
knn_spec <-
  nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification") %>% 
  set_args(neighbors = tune(),
           weight_func = "optimal",
           dist_power = tune())

#bag decision tree rpart
bt_spec <- 
  bag_tree() %>% 
  set_mode("classification") %>% 
  set_engine("rpart") %>% 
  set_args(tree_depth = tune(),
           min_n = tune(),
           cost_complexity = tune())

# logistic
# logreg_spec <- 
#   logistic_reg() %>% 
#   set_mode("classification") %>% 
#   set_engine("glmnet") %>% 
#   set_args(mixture = 1,
#            penalty = tune())

# mars
mars_spec <-
  bag_mars() %>% 
  set_engine("earth") %>% 
  set_mode("classification") %>% 
  set_args(num_terms = tune(), 
           prod_degree = tune())

#svm radial
svm_spec <-
  svm_rbf() %>% 
  set_engine("kernlab") %>% 
  set_mode("classification") %>% 
  set_args(cost = tune(),
           rbf_sigma = tune(),
           margin = tune())

#mlp
mlp_spec <-
  mlp() %>% 
  set_mode("classification") %>% 
  set_engine("nnet") %>% 
  set_args(penalty = tune(),
           epochs = tune(),
           hidden_units = tune())

#lightgbm
lgb_spec <-
  boost_tree(trees = tune(), min_n = tune(), tree_depth = tune(),
             learn_rate = tune(), stop_iter = 10) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

#naive_bayes
nb_spec <-
  naive_Bayes() %>% 
  set_engine("naivebayes") %>% 
  set_mode("classification") %>% 
  set_args(smoothness = tune())

null_spec <-
  null_model() %>% 
  set_mode("classification") %>% 
  set_engine("parsnip")

# workflow set

base_set <- 
  workflow_set (
    list(age_rec = age_rec), #preprocessor
    list(null = null_spec,
         #logistic = logreg_spec,
         knn = knn_spec,
         naivebayes = nb_spec,
         lightgbm = lgb_spec,
         mlp = mlp_spec,
         svm = svm_spec,
         #bag_mars = mars_spec,
         bag_tree = bt_spec,
         xgboost = xgb_spec,
         rand_forest = rf_spec
    ), #model
    cross = TRUE) #default is cross = TRUE

# tune_grid
set.seed(2024060302)
doParallel::registerDoParallel(cl=15, cores = 30)

first_tune_results <-
  workflow_map(base_set,
               fn = "tune_grid",
               verbose = TRUE,
               seed = 2024060302,
               resamples = data_fold,
               metrics = metric_set(roc_auc, accuracy),
               control = control_grid(verbose = TRUE,
                                      allow_par = TRUE,
                                      parallel_over = "everything"))

save(first_tune_results, file = "first_tune_results.Rda")

first_tune_results %>% 
  collect_metrics() %>% 
  filter(.metric == "accuracy") %>% 
  arrange(desc(mean))

autoplot(first_tune_results)

xgboost_res <-
  extract_workflow_set_result(first_tune_results, "age_rec_xgboost")

first_tune_param <-
  extract_workflow_set_result(first_tune_results, "age_rec_xgboost") %>%
  select_best(metric = "accuracy")

# new recipe
inter_rec <-
  recipes::recipe(formula = NObeyesdad ~.,
                  data = data_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_nzv(all_predictors()) %>% # remove zero variance
  step_mutate(age_group = Age) %>% 
  step_cut(age_group,
           breaks = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
           include_outside_range = TRUE) %>%
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_interact(terms = ~ all_numeric_predictors():all_numeric_predictors(),
                role = "predictor") %>% 
  step_dummy(all_nominal_predictors())

inter_set <- 
  workflow_set (
    list(age_rec = age_rec,
         inter_rec = inter_rec), #preprocessor
    list(xgboost = xgb_spec), #model
    cross = TRUE) #default is cross = TRUE

# tune_grid based on xgboost model only
set.seed(2024060303)
doParallel::registerDoParallel(cl=15, cores = 30)

inter_tune_results <-
  workflow_map(inter_set,
               fn = "tune_grid",
               verbose = TRUE,
               seed = 2024060302,
               resamples = data_fold,
               metrics = metric_set(roc_auc, accuracy),
               control = control_grid(verbose = TRUE,
                                      allow_par = TRUE,
                                      parallel_over = "everything"))

save(inter_tune_results, file = "inter_tune_results.Rda")

inter_tune_results %>% 
  collect_metrics() %>% 
  filter(.metric == "accuracy") %>% 
  arrange(desc(mean))

autoplot(inter_tune_results)

# interactions did not improve results

# second round eval using tune_sim_anneal
# xgboost wflow
xgboost_wflow <-
  workflow() %>% 
  add_model(xgb_spec) %>% 
  add_recipe(age_rec)
  
# set up new search grid
xgboost_grid <-
  extract_parameter_set_dials(xgb_spec) %>% 
  update(mtry = mtry(c(2L,10L)),
         trees = trees(c(800L, 1000L)),
         min_n = min_n(c(15L,30L)),
         tree_depth = tree_depth(c(5L,15L)),
         learn_rate = learn_rate(c(0.02, 0.5)),
         loss_reduction = loss_reduction(c(0.00002, 0.00005)),
         sample_size = sample_size(c(0,1L))
  )

xgboost_metrics <-
  metric_set(accuracy)

# tune_sim_anneal

set.seed(2024040301)
doParallel::registerDoParallel(cl=15, cores = 30)

age_xgboost_sim_anneal_result <-
  tune_sim_anneal(
    object = xgboost_wflow,
    resamples = data_fold,
    iter = 50,
    metrics = xgboost_metrics,
    param_info = xgboost_grid,
    initial = 1,
    control = control_sim_anneal(verbose = TRUE,
                                 verbose_iter = TRUE,
                                 allow_par = TRUE,
                                 parallel_over = "everything")
  )

save(age_xgboost_sim_anneal_result, file = "age_xgboost_sim_anneal_result.Rda")

# no improvement, try poly recipe
poly_rec <-
  recipes::recipe(formula = NObeyesdad ~.,
                  data = data_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_nzv(all_predictors()) %>% # remove zero variance
  step_poly(all_numeric_predictors(), degree = 2) %>% 
  step_YeoJohnson(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

xgboost_wflow <-
  workflow() %>% 
  add_model(xgb_spec) %>% 
  add_recipe(poly_rec)


set.seed(2024040304)
doParallel::registerDoParallel(cl=15, cores = 30)
poly_xgboost_result <-
  xgboost_wflow %>% 
  tune_grid(resamples = data_fold,
            metrics = xgboost_metrics
  )

save(poly_xgboost_result, file = "poly_xgboost_result.Rda")

poly_xgboost_result %>%
  collect_metrics() %>% 
  select_best()
  filter(.metric == "accuracy") %>% 
  arrange(desc(mean))

# fit and compare
data_test <- read_csv("test.csv")

age_xgboost_wflow <-
  workflow() %>% 
  add_model(xgb_spec) %>% 
  add_recipe(age_rec)

age_xgboost_tuned_fit_wflow <-
  age_xgboost_wflow %>% 
  finalize_workflow(first_tune_param) %>% 
  fit(data_train)

results_age_xgboost <-
  age_xgboost_tuned_fit_wflow %>% 
  predict(data_test)

  