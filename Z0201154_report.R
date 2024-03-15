library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3viz)
library(data.table)
library(ggplot2)
library(precrec)

# Load the dataset from a URL
data <- fread("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")

# Adjust the target variable to a factor type
data[, fatal_mi := as.factor(fatal_mi)]

# Display a summary of the dataset
print(summary(data))

# Visualizing the distribution of the target variable 'fatal_mi'
ggplot(data, aes(fatal_mi)) +
  geom_bar(fill = "cornflowerblue", color = "black") +
  labs(title = "Distribution of Fatal MI", x = "Fatal MI Status", y = "Frequency")

# Identify and visualize numerical features, excluding 'fatal_mi'
numeric_features <- names(data)[sapply(data, is.numeric) & names(data) != "fatal_mi"]

# Improved visualization of distributions
ggplot(data, aes(x = factor(0))) +
  geom_histogram(data = melt(data, measure.vars = numeric_features), aes(x = value, fill = variable), position = "dodge", bins = 30) +
  facet_wrap(~variable, scales = "free", ncol = 2) +
  labs(x = "Value", y = "Frequency") +
  theme_minimal() +
  theme(legend.title = element_blank())

# Define classification task
task <- TaskClassif$new(id = "FatalMIPrediction", backend = data, target = "fatal_mi")

# Select learners
learners <- list(
  lrn("classif.log_reg", predict_type = "prob"),
  lrn("classif.svm", predict_type = "prob"),
  lrn("classif.ranger", predict_type = "prob")
)

# Cross-validation
resampling <- rsmp("cv", folds = 5)

# Benchmarking
design <- benchmark_grid(
  tasks = task,
  learners = learners,
  resamplings = resampling
)

# Check if design is a data frame
if (!is.data.frame(design)) {
  stop("The design is not a data frame structure.")
}

# Benchmark models
bmr <- benchmark(design)

# Aggregate results
bmr_results <- bmr$aggregate(msr("classif.auc"))
autoplot(bmr, measure = msr("classif.auc"))

# RandomForest hyperparameter tuning
learner_rf <- lrn("classif.ranger", predict_type = "prob")
param_set <- ParamSet$new(params = list(
  ParamInt$new("mtry", lower = as.integer(sqrt(ncol(data)/3)), upper = as.integer(sqrt(ncol(data)))),
  ParamInt$new("min.node.size", lower = 1, upper = 10),
  ParamInt$new("num.trees", lower = 100, upper = 1000)
))

tuner <- tnr("random_search", batch_size = 10)
at <- AutoTuner$new(
  learner = learner_rf,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  tuner = tuner,
  search_space = param_set,
  terminator = trm("evals", n_evals = 50)
)

# Train the tuned model
at$train(task)

# Make predictions and plot ROC curve
prediction <- at$predict(task)
autoplot(prediction, type = "roc")

# Print confusion matrix
conf_mat <- prediction$confusion
print(conf_mat)