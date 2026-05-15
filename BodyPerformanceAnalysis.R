#  TEAM WORKFLOW ORDER:
#  ├── PART 1 : Data Loading, Preprocessing & EDA    
#  ├── PART 2 : Statistical Analysis & Hypothesis    
#  ├── PART 3 : Tree-Based ML Models                
#  ├── PART 4 : Logistic Regression & KNN           
#  └── PART 5 : k-means clustring                    
#

# ================================================================
# PACKAGE INSTALLATION
# ================================================================

install.packages("corrplot")
install.packages("party")
install.packages("randomForest")
install.packages("e1071")
install.packages("xgboost")
install.packages("caret")
install.packages("class")
install.packages("MLmetrics")
install.packages("nnet")


# ================================================================
# PART 1 : DATA LOADING, PREPROCESSING & EDA
# ================================================================


# ────────────────────────────────────────────────────────────────
# 1.1  LOAD DATASET
# ────────────────────────────────────────────────────────────────

data <- read.csv("C:/Users/Menna Thabet/Downloads/bodyPerformance.csv")


# ────────────────────────────────────────────────────────────────
# 1.2  EXPLORE DATASET
# ────────────────────────────────────────────────────────────────

head(data)

str(data)

summary(data)

dim(data)

names(data)


# ────────────────────────────────────────────────────────────────
# 1.3  CHECK MISSING VALUES
# ────────────────────────────────────────────────────────────────

sum(is.na(data))

colSums(is.na(data))


# ────────────────────────────────────────────────────────────────
# 1.4  CHECK & REMOVE DUPLICATE ROWS
# ────────────────────────────────────────────────────────────────

sum(duplicated(data))

data <- unique(data)


# ────────────────────────────────────────────────────────────────
# 1.5  ENCODE CATEGORICAL VARIABLES
# ────────────────────────────────────────────────────────────────

data$gender <- as.factor(data$gender)

data$class <- as.factor(data$class)

# Check Dataset Structure Again

str(data)

summary(data)


# ────────────────────────────────────────────────────────────────
# 1.7  REMOVE IMPOSSIBLE / INVALID VALUES
# ────────────────────────────────────────────────────────────────

# Remove impossible blood pressure values

data <- subset(data, systolic > 0)

data <- subset(data, diastolic > 0)

# Remove impossible grip force values

data <- subset(data, gripForce > 0)


# ────────────────────────────────────────────────────────────────
# 1.8  NORMALIZE NUMERICAL DATA
# ────────────────────────────────────────────────────────────────

normalize <- function(x)
{
  return((x - min(x)) / (max(x) - min(x)))
}

data$age_norm <- normalize(data$age)

data$weight_norm <- normalize(data$weight_kg)

data$bodyfat_norm <- normalize(data$body.fat_.)

data$grip_norm <- normalize(data$gripForce)


# ────────────────────────────────────────────────────────────────
# 1.9  EXPLORATORY DATA ANALYSIS (EDA)
# ────────────────────────────────────────────────────────────────

# --- Combined boxplots ---

par(mfrow = c(2,2))

boxplot(data$age,
        col = "lightblue",
        main = "Age Boxplot",
        ylab = "Age")

boxplot(data$weight_kg,
        col = "lightgreen",
        main = "Weight Boxplot",
        ylab = "Weight")

boxplot(data$body.fat_.,
        col = "pink",
        main = "Body Fat Boxplot",
        ylab = "Body Fat %")

boxplot(data$gripForce,
        col = "orange",
        main = "Grip Force Boxplot",
        ylab = "Grip Force")

rug(data$gripForce, side = 2)


# --- Chart 1 : Class Distribution ---

counts <- table(data$class)

barplot(counts,
        col = rainbow(4),
        main = "Class Distribution",
        xlab = "Class",
        ylab = "Count")


# --- Chart 2 : Gender Distribution ---

gender_count <- table(data$gender)

pie(gender_count,
    labels = gender_count,
    col = rainbow(length(gender_count)),
    main = "Gender Distribution")


# --- Chart 3 : Age Distribution ---

hist(data$age,
     col = "lightblue",
     breaks = 10,
     main = "Age Distribution",
     xlab = "Age")


# --- Chart 4 : Weight Distribution ---

hist(data$weight_kg,
     col = "lightgreen",
     breaks = 10,
     main = "Weight Distribution",
     xlab = "Weight")


# --- Chart 5 : Body Fat Distribution ---

hist(data$body.fat_.,
     prob = TRUE,
     col = "grey",
     main = "Body Fat Distribution",
     xlab = "Body Fat %")

lines(density(data$body.fat_.),
      col = "blue",
      lwd = 2)


# --- Chart 6 : Grip Force Distribution ---

hist(data$gripForce,
     col = "orange",
     breaks = 10,
     main = "Grip Force Distribution",
     xlab = "Grip Force")


# --- Chart 7 : Body Fat vs Class ---

boxplot(body.fat_. ~ class,
        data = data,
        col = rainbow(4),
        main = "Body Fat Percentage by Class",
        xlab = "Class",
        ylab = "Body Fat %")


# --- Chart 8 : Grip Force by Gender ---

boxplot(gripForce ~ gender,
        data = data,
        col = c("lightblue", "pink"),
        main = "Grip Force by Gender",
        xlab = "Gender",
        ylab = "Grip Force")


# --- Chart 9 : Sit-Ups by Class ---

boxplot(sit.ups.counts ~ class,
        data = data,
        col = rainbow(4),
        main = "Sit-Ups Counts by Class",
        xlab = "Class",
        ylab = "Sit-Ups Counts")


# --- Chart 10 : Broad Jump by Class ---

boxplot(broad.jump_cm ~ class,
        data = data,
        col = rainbow(4),
        main = "Broad Jump by Class",
        xlab = "Class",
        ylab = "Broad Jump")


# --- Chart 11 : Age vs Sit-Ups ---

plot(data$age,
     data$sit.ups.counts,
     col = "blue",
     pch = 19,
     main = "Age vs Sit-Ups Counts",
     xlab = "Age",
     ylab = "Sit-Ups Counts")


# --- Chart 12 : Grip Force vs Broad Jump ---

plot(data$gripForce,
     data$broad.jump_cm,
     col = "red",
     pch = 19,
     main = "Grip Force vs Broad Jump",
     xlab = "Grip Force",
     ylab = "Broad Jump")


# --- Chart 13 : Height vs Weight ---

plot(data$height_cm,
     data$weight_kg,
     col = "darkgreen",
     pch = 19,
     main = "Height vs Weight",
     xlab = "Height (cm)",
     ylab = "Weight (kg)")


# --- Chart 14 : Systolic vs Diastolic Blood Pressure ---

plot(data$systolic,
     data$diastolic,
     col = "purple",
     pch = 19,
     main = "Systolic vs Diastolic Pressure",
     xlab = "Systolic",
     ylab = "Diastolic")


# --- Chart 15 : Grip Force Dot Plot ---

dotchart(data$gripForce,
         labels = row.names(data),
         cex = 0.5,
         main = "Grip Force Dot Plot",
         xlab = "Grip Force")


# --- Chart 16 : Correlation Matrix & Plot ---

numeric_data <- data[, sapply(data, is.numeric)]

cor_matrix <- cor(numeric_data)

cor_matrix

library(corrplot)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.cex = 0.7)


# --- Chart 17 : Scatter Plot Matrix ---

dev.off()

graphics.off()

pairs(numeric_data)

# ────────────────────────────────────────────────────────────────
# 1.10  EXPORT CLEAN DATASET
# ────────────────────────────────────────────────────────────────

write.csv(data,
          "cleaned_bodyPerformance.csv",
          row.names = FALSE)

# ── END OF PART 1 ───────────────────────────────────────────────



# ================================================================
# PART 2 : STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# ================================================================


# ────────────────────────────────────────────────────────────────
# 2.1  DESCRIPTIVE STATISTICS
# ────────────────────────────────────────────────────────────────

summary(data)

mean(data$gripForce)
median(data$gripForce)
sd(data$gripForce)

mean(data$body.fat..)
median(data$body.fat..)
sd(data$body.fat..)

mean(data$sit.ups.counts)
median(data$sit.ups.counts)
sd(data$sit.ups.counts)

mean(data$broad.jump_cm)
median(data$broad.jump_cm)
sd(data$broad.jump_cm)


# ────────────────────────────────────────────────────────────────
# 2.2  HYPOTHESIS 1 — INDEPENDENT T-TEST
#       Research Question : Does gender affect grip strength?
#       H0 : No significant difference in grip strength by gender
#       H1 : Significant difference exists
# ────────────────────────────────────────────────────────────────

t_test_result <- t.test(gripForce ~ gender, data = data)

t_test_result


# ────────────────────────────────────────────────────────────────
# 2.3  CORRELATION TEST
#       Variables : Grip Force & Broad Jump
# ────────────────────────────────────────────────────────────────

correlation_result <- cor.test(data$gripForce,
                               data$broad.jump_cm)

correlation_result


# ────────────────────────────────────────────────────────────────
# 2.4  HYPOTHESIS 2 — ANOVA TEST
#       Research Question : Does body fat % differ across classes?
#       H0 : Mean body fat % is equal across all classes
#       H1 : At least one class has a different mean body fat %
# ────────────────────────────────────────────────────────────────

anova_result <- aov(sit.ups.counts ~ class,
                    data = data)

summary(anova_result)


# ────────────────────────────────────────────────────────────────
# 2.5  CHI-SQUARE TEST
#       Variables : Gender & Class
# ────────────────────────────────────────────────────────────────

table_gender_class <- table(data$gender,
                            data$class)

chi_square_result <- chisq.test(table_gender_class)

chi_square_result

# ── END OF PART 2 ───────────────────────────────────────────────




# ================================================================
# PART 3 : TREE-BASED ML MODELS
# Models   : Decision Tree · Random Forest · XGBoost · Naive Bayes
# ================================================================


# ────────────────────────────────────────────────────────────────
# 3.1  LOAD LIBRARIES
# ────────────────────────────────────────────────────────────────

library(party)
library(randomForest)
library(e1071)
library(xgboost)
library(caret)


# ────────────────────────────────────────────────────────────────
# 3.2  LOAD CLEANED DATASET
# ────────────────────────────────────────────────────────────────

bodyData <- read.csv("C:/Users/Menna Thabet/Downloads/cleaned_bodyPerformance.csv")
bodyData

table(bodyData$class)

bodyData$class  <- as.factor(bodyData$class)
bodyData$gender <- as.factor(bodyData$gender)

# Remove pre-normalised columns
bodyData <- bodyData[ , !names(bodyData) %in%
                        c("age_norm", "weight_norm", "bodyfat_norm", "grip_norm")]


# ────────────────────────────────────────────────────────────────
# 3.3  ADDITIONAL EXPLORATORY PLOTS
# ────────────────────────────────────────────────────────────────

# Body fat histogram
png("C:\\Users\\Menna Thabet\\Downloads\tree_based_models\\tree_based_models\\plot_bodyfat.png")
par(mfcol = c(1,1))
hist(bodyData$body.fat_., col = "lightblue", xlab = "Body Fat (%)", main = "Body Fat Distribution")
dev.off()

# Grip force histogram
png("C:/Users/Menna Thabet/Downloads/tree_based_models/tree_based_models/plot_gripforce.png")
hist(bodyData$gripForce, col = "lightblue", xlab = "Grip Force",
     main = "Grip Force Distribution", breaks = seq(0, 80, by = 5))
dev.off()

# Weight in multiple units
png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_weight_units.png")
par(mfcol = c(3,1))
hist(bodyData$weight_kg * 1000, breaks = 10, main = "Weight (in g)",      xlab = "Weight")
hist(bodyData$weight_kg,         breaks = 10, main = "Weight (in kg)",     xlab = "Weight")
hist(bodyData$weight_kg / 1000,  breaks = 10, main = "Weight (in tonnes)", xlab = "Weight")
dev.off()

# Boxplot of weight by class
png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_weight_boxplot.png")
par(mfcol = c(1,1))
boxplot(weight_kg ~ class, data = bodyData, col = rainbow(4),
        ylab = "Weight (kg)", xlab = "Performance Class", main = "Weight by Performance Class")
rug(bodyData$weight_kg, side = 2)
dev.off()

# Scatter: grip force vs sit-ups
png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_scatter.png")
plot(bodyData$gripForce, bodyData$sit.ups.counts, col = as.integer(bodyData$class),
     pch = 16, xlab = "Grip Force", ylab = "Sit-Up Counts", main = "Grip Force vs Sit-Up Counts")
legend("topright", legend = levels(bodyData$class), col = 1:4, pch = 16, title = "Class")
dev.off()

# Average sit-ups per class
AvgSitupsPerClass <- aggregate(bodyData[ , "sit.ups.counts"], list(bodyData$class), mean)
AvgSitupsPerClass
png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_avg_situps.png")
barplot(AvgSitupsPerClass$x, names.arg = AvgSitupsPerClass$Group.1, col = rainbow(4),
        main = "Average Sit-Ups per Performance Class", xlab = "Class", ylab = "Mean Sit-Up Count")
dev.off()

# Broad jump density plot
png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_broadjump_density.png")
hist(bodyData$broad.jump_cm, prob = TRUE, col = "grey",
     xlab = "Broad Jump (cm)", main = "Broad Jump Distribution")
lines(density(bodyData$broad.jump_cm), col = "blue", lwd = 2)
lines(density(bodyData$broad.jump_cm, adjust = 2), lty = "dotted", col = "darkgreen", lwd = 2)
legend("topright", legend = c("KDE (default)", "KDE (smoother)"),
       col = c("blue", "darkgreen"), lty = c("solid", "dotted"), lwd = 2)
dev.off()

# Dot chart of mean feature values per class
classMeans <- aggregate(. ~ class,
                        data = bodyData[ , c("class", "gripForce", "sit.ups.counts",
                                             "broad.jump_cm", "body.fat_.")],
                        FUN  = mean)
png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_dotchart.png")
dotchart(as.matrix(classMeans[ , -1]), labels = classMeans$class, cex = 0.8,
         main = "Mean Feature Values per Class", xlab = "Mean Value")
dev.off()


# ────────────────────────────────────────────────────────────────
# 3.4  TRAIN / TEST SPLIT  (§g)
# ────────────────────────────────────────────────────────────────

ind        <- sample(2, nrow(bodyData), prob = c(0.7, 0.3), replace = TRUE)
train.data <- bodyData[ind == 1, ]
test.data  <- bodyData[ind == 2, ]


# ────────────────────────────────────────────────────────────────
# 3.5  MODEL 1 — DECISION TREE  (§h)
# ────────────────────────────────────────────────────────────────

body.tree <- ctree(
  class ~ age + gender + height_cm + weight_kg + body.fat_. +
          diastolic + systolic + gripForce +
          sit.and.bend.forward_cm + sit.ups.counts + broad.jump_cm,
  data = train.data
)

png("C:/Users/Menna Thabet/Downloads/tree_based_models/tree_based_models/plot_decision_tree.png", width = 1400, height = 900)
plot(body.tree, type = "simple")
dev.off()

testPred <- predict(body.tree, newdata = test.data)
table(testPred, test.data$class)

dt.acc <- mean(testPred == test.data$class)
cat(sprintf("Decision Tree Accuracy: %.4f\n", dt.acc))


# ────────────────────────────────────────────────────────────────
# 3.6  MODEL 2 — RANDOM FOREST  (§h)
# ────────────────────────────────────────────────────────────────

rf.model <- randomForest(class ~ .,
                         data       = train.data,
                         ntree      = 500,
                         mtry       = round(sqrt(ncol(train.data) - 1)),
                         importance = TRUE)
rf.model

rf.pred <- predict(rf.model, newdata = test.data)
tab.rf  <- table(Predicted = rf.pred, Actual = test.data$class)
tab.rf

rf.acc <- mean(rf.pred == test.data$class)
cat(sprintf("Random Forest Accuracy: %.4f\n", rf.acc))

png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_rf_importance.png")
varImpPlot(rf.model, main = "Random Forest - Variable Importance", col = "steelblue")
dev.off()


# ────────────────────────────────────────────────────────────────
# 3.7  MODEL 3 — NAIVE BAYES  (§h)
# ────────────────────────────────────────────────────────────────

classifier <- naiveBayes(class ~ ., data = train.data)

Output <- predict(classifier, test.data)
Tab    <- table(Output, test.data$class)
Tab

nb.acc <- mean(Output == test.data$class)
cat(sprintf("Naive Bayes Accuracy: %.4f\n", nb.acc))


# ────────────────────────────────────────────────────────────────
# 3.8  MODEL 4 — XGBOOST  (§h)
# ────────────────────────────────────────────────────────────────

# Encode class as numeric (0-based)
train.label <- as.integer(train.data$class) - 1
test.label  <- as.integer(test.data$class) - 1

# One-hot encode features
train.matrix <- model.matrix(class ~ . - 1, data = train.data)
test.matrix  <- model.matrix(class ~ . - 1, data = test.data)

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = train.matrix, label = train.label)
dtest  <- xgb.DMatrix(data = test.matrix,  label = test.label)

# Parameters
params <- list(
  objective = "multi:softmax",
  num_class = 4,
  max_depth = 6,
  eta = 0.1,
  eval_metric = "merror"
)

# Train model
xgb.model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

# Predictions
xgb.pred.num <- predict(xgb.model, dtest)

# Convert back to class labels
xgb.pred <- levels(train.data$class)[xgb.pred.num + 1]

# Confusion matrix
tab.xgb <- table(
  Predicted = xgb.pred,
  Actual = test.data$class
)

tab.xgb

# Accuracy
xgb.acc <- mean(xgb.pred == test.data$class)

cat(sprintf("XGBoost Accuracy: %.4f\n", xgb.acc))

# Variable importance
xgb.imp <- xgb.importance(
  feature_names = colnames(train.matrix),
  model = xgb.model
)

xgb.imp

# ────────────────────────────────────────────────────────────────
# 3.9  K-MEANS CLUSTERING  (§h unsupervised)
# ────────────────────────────────────────────────────────────────

newBody        <- bodyData
newBody$class  <- NULL
newBody$gender <- NULL
newBody

kc <- kmeans(newBody, 4)
kc

kc$cluster
kc$centers

table(bodyData$class, kc$cluster)

png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_kmeans.png")
plot(newBody$gripForce, newBody$sit.ups.counts, col = kc$cluster,
     xlab = "Grip Force", ylab = "Sit-Up Counts", main = "K-Means Clusters (k=4)")
points(kc$centers[ , c("gripForce", "sit.ups.counts")], col = 1:4, pch = 8, cex = 2)
dev.off()


# ────────────────────────────────────────────────────────────────
# 3.10  MODEL COMPARISON — TREE-BASED MODELS  (§i)
# ────────────────────────────────────────────────────────────────

acc.results <- c(
  DecisionTree = dt.acc,
  RandomForest = rf.acc,
  XGBoost      = xgb.acc,
  NaiveBayes   = nb.acc
)

print(round(acc.results, 4))

png("C:\Users\Menna Thabet\Downloads\tree_based_models\tree_based_models\plot_model_comparison.png")
barplot(acc.results, col = rainbow(4), ylim = c(0, 1),
        main = "Test-Set Accuracy by Model", ylab = "Accuracy",
        xlab = "Model", names.arg = names(acc.results))
text(x = c(0.7, 1.9, 3.1, 4.3), y = acc.results + 0.03, labels = round(acc.results, 3), cex = 0.9)
dev.off()

cat("Best model:", names(which.max(acc.results)), "\n")

# ── END OF PART 3 ───────────────────────────────────────────────




# ================================================================
# PART 4 : LOGISTIC REGRESSION
# ================================================================


# ────────────────────────────────────────────────────────────────
# 4.1  LOAD LIBRARIES
# ────────────────────────────────────────────────────────────────

library(caret)
library(nnet)
library(MLmetrics)


# ────────────────────────────────────────────────────────────────
# 4.2  LOAD DATASET
# ────────────────────────────────────────────────────────────────

data <- read.csv("C:/Users/Menna Thabet/Downloads/cleaned_bodyPerformance.csv")


# ────────────────────────────────────────────────────────────────
# 4.3  REMOVE REDUNDANT COLUMNS
# ────────────────────────────────────────────────────────────────

data$age_norm <- NULL
data$weight_norm <- NULL
data$bodyfat_norm <- NULL
data$grip_norm <- NULL


# ────────────────────────────────────────────────────────────────
# 4.4  CONVERT VARIABLES
# ────────────────────────────────────────────────────────────────

data$class <- as.factor(data$class)
data$gender <- as.factor(data$gender)


# ────────────────────────────────────────────────────────────────
# 4.5  TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────

set.seed(123)
trainIndex <- createDataPartition(
  data$class,
  p = 0.8,
  list = FALSE
)

trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]


# ────────────────────────────────────────────────────────────────
# 4.6  NORMALIZE NUMERIC FEATURES
# ────────────────────────────────────────────────────────────────

# Create dummy variables
dummies <- dummyVars(class ~ ., data = trainData)

trainX <- predict(dummies, newdata = trainData)
testX  <- predict(dummies, newdata = testData)

# Scale features
preProcValues <- preProcess(
  trainX,
  method = c("center", "scale")
)

trainX_scaled <- predict(preProcValues, trainX)
testX_scaled  <- predict(preProcValues, testX)

# Add target back
trainScaled <- data.frame(
  trainX_scaled,
  class = trainData$class
)
testScaled <- data.frame(
  testX_scaled,
  class = testData$class
)


# ────────────────────────────────────────────────────────────────
# 4.7  TRAIN LOGISTIC REGRESSION MODEL
#       Multinomial Logistic Regression
# ────────────────────────────────────────────────────────────────

model <- multinom(
  class ~ .,
  data = trainScaled
)


# ────────────────────────────────────────────────────────────────
# 4.8  MODEL SUMMARY
# ────────────────────────────────────────────────────────────────

summary(model)


# ────────────────────────────────────────────────────────────────
# 4.9  PREDICTIONS
# ────────────────────────────────────────────────────────────────

predictions <- predict(
  model,
  newdata = testScaled
)


# ────────────────────────────────────────────────────────────────
# 4.10  CONFUSION MATRIX  (§i)
# ────────────────────────────────────────────────────────────────

cm <- confusionMatrix(
  predictions,
  testScaled$class
)

cat("\n===== Confusion Matrix =====\n")
print(cm)


# ────────────────────────────────────────────────────────────────
# 4.11  ACCURACY  (§i)
# ────────────────────────────────────────────────────────────────

accuracy <- cm$overall["Accuracy"]

cat("\nAccuracy:",
    round(accuracy * 100, 2),
    "%\n")


# ────────────────────────────────────────────────────────────────
# 4.12  PRECISION · RECALL · F1-SCORE  (§i)
# ────────────────────────────────────────────────────────────────

precision <- mean(
  cm$byClass[, "Precision"],
  na.rm = TRUE
)

recall <- mean(
  cm$byClass[, "Recall"],
  na.rm = TRUE
)

f1_score <- mean(
  cm$byClass[, "F1"],
  na.rm = TRUE
)

cat("\nPrecision:",
    round(precision, 4),
    "\n")

cat("Recall:",
    round(recall, 4),
    "\n")

cat("F1-Score:",
    round(f1_score, 4),
    "\n")


# ────────────────────────────────────────────────────────────────
# 4.13  ANALYZE COEFFICIENTS
# ────────────────────────────────────────────────────────────────

cat("\n===== Model Coefficients =====\n")
print(summary(model)$coefficients)

# ── END OF PART 4 ───────────────────────────────────────────────




# ================================================================
# PART 5 : KNN MODEL
# ================================================================


# ────────────────────────────────────────────────────────────────
# 5.1  LOAD LIBRARIES
# ────────────────────────────────────────────────────────────────

library(caret)
library(class)
library(MLmetrics)


# ────────────────────────────────────────────────────────────────
# 5.2  LOAD DATASET
# ────────────────────────────────────────────────────────────────

data <- read.csv("C:/Users/Menna Thabet/Downloads/cleaned_bodyPerformance.csv")


# ────────────────────────────────────────────────────────────────
# 5.3  REMOVE REDUNDANT COLUMNS
# ────────────────────────────────────────────────────────────────

data$age_norm <- NULL
data$weight_norm <- NULL
data$bodyfat_norm <- NULL
data$grip_norm <- NULL


# ────────────────────────────────────────────────────────────────
# 5.4  CONVERT VARIABLES
# ────────────────────────────────────────────────────────────────

data$class <- as.factor(data$class)
data$gender <- as.factor(data$gender)


# ────────────────────────────────────────────────────────────────
# 5.5  TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────

set.seed(123)

trainIndex <- createDataPartition(
  data$class,
  p = 0.8,
  list = FALSE
)

trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]


# ────────────────────────────────────────────────────────────────
# 5.6  SEPARATE FEATURES AND LABELS
# ────────────────────────────────────────────────────────────────

trainX <- trainData[, !(names(trainData) %in% "class")]
testX  <- testData[, !(names(testData) %in% "class")]

trainY <- trainData$class
testY  <- testData$class


# ────────────────────────────────────────────────────────────────
# 5.7  ONE-HOT ENCODING
# ────────────────────────────────────────────────────────────────

dummies <- dummyVars(~ ., data = trainX)

trainX <- predict(dummies, newdata = trainX)
testX  <- predict(dummies, newdata = testX)


# ────────────────────────────────────────────────────────────────
# 5.8  NORMALIZE FEATURES
# ────────────────────────────────────────────────────────────────

preProcValues <- preProcess(
  trainX,
  method = c("center", "scale")
)

trainX_scaled <- predict(preProcValues, trainX)
testX_scaled  <- predict(preProcValues, testX)


# ────────────────────────────────────────────────────────────────
# 5.9  FIND BEST K  (§h)
# ────────────────────────────────────────────────────────────────

k_values <- seq(1, 31, 2)

accuracy_values <- c()

for (k in k_values) {

  predictions <- knn(
    train = trainX_scaled,
    test = testX_scaled,
    cl = trainY,
    k = k
  )

  accuracy <- mean(predictions == testY)

  accuracy_values <- c(
    accuracy_values,
    accuracy
  )

  cat("K =", k,
      " Accuracy =",
      round(accuracy * 100, 2),
      "%\n")
}

best_k <- k_values[which.max(accuracy_values)]

cat("\nBest K =", best_k, "\n")


# ────────────────────────────────────────────────────────────────
# 5.10  FINAL KNN PREDICTIONS
# ────────────────────────────────────────────────────────────────

final_predictions <- knn(
  train = trainX_scaled,
  test = testX_scaled,
  cl = trainY,
  k = best_k
)


# ────────────────────────────────────────────────────────────────
# 5.11  CONFUSION MATRIX  (§i)
# ────────────────────────────────────────────────────────────────

cm <- confusionMatrix(
  final_predictions,
  testY
)

cat("\n===== Confusion Matrix =====\n")
print(cm)


# ────────────────────────────────────────────────────────────────
# 5.12  ACCURACY  (§i)
# ────────────────────────────────────────────────────────────────

accuracy <- cm$overall["Accuracy"]

cat("\nAccuracy:",
    round(accuracy * 100, 2),
    "%\n")


# ────────────────────────────────────────────────────────────────
# 5.13  PRECISION · RECALL · F1-SCORE  (§i)
# ────────────────────────────────────────────────────────────────

# For multiclass:
# Calculate weighted averages

precision <- mean(cm$byClass[, "Precision"], na.rm = TRUE)

recall <- mean(cm$byClass[, "Recall"], na.rm = TRUE)

f1_score <- mean(cm$byClass[, "F1"], na.rm = TRUE)

cat("\nPrecision:",
    round(precision, 4),
    "\n")

cat("Recall:",
    round(recall, 4),
    "\n")

cat("F1-Score:",
    round(f1_score, 4),
    "\n")


# ────────────────────────────────────────────────────────────────
# 5.14  PLOT ACCURACY VS K  (§i)
# ────────────────────────────────────────────────────────────────

plot(
  k_values,
  accuracy_values,
  type = "b",
  xlab = "K Value",
  ylab = "Accuracy",
  main = "KNN Accuracy vs K"
)

# ── END OF PART 5 ───────────────────────────────────────────────




# ================================================================
# PART 6 : CLUSTERING
# ================================================================


# ────────────────────────────────────────────────────────────────
# 6.1  LOAD DATASET
# ────────────────────────────────────────────────────────────────

data <- read.csv("C:/Users/Menna Thabet/Downloads/cleaned_bodyPerformance.csv")

head(data)
str(data)
summary(data)


# ================================================================
# SECTION 1: K-MEANS CLUSTERING
# ================================================================

# We select numeric performance-related features for clustering
# (excluding the normalized columns and non-performance variables)

cluster_data <- data[, c("gripForce", "sit.ups.counts", "broad.jump_cm", "body.fat_.")]

head(cluster_data)


# ────────────────────────────────────────────────────────────────
# 6.2  DETERMINE OPTIMAL NUMBER OF CLUSTERS — ELBOW METHOD  (§h)
# ────────────────────────────────────────────────────────────────

wss <- c()

for (k in 1:10) {
  km_temp <- kmeans(cluster_data, centers = k, nstart = 10)
  wss[k] <- km_temp$tot.withinss
}

plot(1:10, wss,
     type = "b",
     pch = 19,
     col = "blue",
     main = "Elbow Method - Optimal K",
     xlab = "Number of Clusters (K)",
     ylab = "Total Within-Cluster Sum of Squares")


# Based on the elbow plot, K=4 is chosen
# (matches the 4 performance classes: A, B, C, D)


# ────────────────────────────────────────────────────────────────
# 6.3  APPLY K-MEANS WITH K = 4  (§h)
# ────────────────────────────────────────────────────────────────

set.seed(42)

kc <- kmeans(cluster_data, centers = 4, nstart = 20)

kc


# ────────────────────────────────────────────────────────────────
# 6.4  VIEW CLUSTER ASSIGNMENTS
# ────────────────────────────────────────────────────────────────

kc$cluster

kc$centers

kc$size


# ────────────────────────────────────────────────────────────────
# 6.5  ADD CLUSTER LABELS TO DATASET
# ────────────────────────────────────────────────────────────────

data$cluster <- as.factor(kc$cluster)


# ────────────────────────────────────────────────────────────────
# 6.6  COMPARE CLUSTERS WITH ORIGINAL CLASS LABELS  (§i)
# ────────────────────────────────────────────────────────────────

table(data$class, kc$cluster)


# ────────────────────────────────────────────────────────────────
# 6.7  VISUALIZE CLUSTERS  (§i)
# ────────────────────────────────────────────────────────────────

# Plot 1: Grip Force vs Broad Jump colored by cluster

plot(cluster_data$gripForce, cluster_data$broad.jump_cm,
     col = kc$cluster,
     pch = 19,
     main = "K-Means Clusters: Grip Force vs Broad Jump",
     xlab = "Grip Force",
     ylab = "Broad Jump (cm)")

points(kc$centers[, "gripForce"],
       kc$centers[, "broad.jump_cm"],
       col = 1:4,
       pch = 8,
       cex = 3)

legend("topleft",
       legend = paste("Cluster", 1:4),
       col = 1:4,
       pch = 19,
       cex = 0.8)


# Plot 2: Sit-Ups vs Body Fat colored by cluster

plot(cluster_data$sit.ups.counts, cluster_data$body.fat_.,
     col = kc$cluster,
     pch = 19,
     main = "K-Means Clusters: Sit-Ups vs Body Fat",
     xlab = "Sit-Ups Counts",
     ylab = "Body Fat %")

points(kc$centers[, "sit.ups.counts"],
       kc$centers[, "body.fat_."],
       col = 1:4,
       pch = 8,
       cex = 3)

legend("topright",
       legend = paste("Cluster", 1:4),
       col = 1:4,
       pch = 19,
       cex = 0.8)


# Plot 3: Cluster Centers Bar Chart (Average Feature Values per Cluster)

par(mfrow = c(2, 2))

barplot(kc$centers[, "gripForce"],
        names.arg = paste("C", 1:4),
        col = rainbow(4),
        main = "Avg Grip Force per Cluster",
        ylab = "Grip Force")

barplot(kc$centers[, "sit.ups.counts"],
        names.arg = paste("C", 1:4),
        col = rainbow(4),
        main = "Avg Sit-Ups per Cluster",
        ylab = "Sit-Ups")

barplot(kc$centers[, "broad.jump_cm"],
        names.arg = paste("C", 1:4),
        col = rainbow(4),
        main = "Avg Broad Jump per Cluster",
        ylab = "Broad Jump (cm)")

barplot(kc$centers[, "body.fat_."],
        names.arg = paste("C", 1:4),
        col = rainbow(4),
        main = "Avg Body Fat per Cluster",
        ylab = "Body Fat %")

dev.off()


# ================================================================
# SECTION 2: FINAL INTEGRATION
# ================================================================

# Combine all analytical results into one consolidated summary


# ────────────────────────────────────────────────────────────────
# 6.8  SUMMARY STATISTICS PER CLUSTER  (§i)
# ────────────────────────────────────────────────────────────────

aggregate(cluster_data, by = list(Cluster = kc$cluster), FUN = mean)


# ────────────────────────────────────────────────────────────────
# 6.9  CLASS DISTRIBUTION PER CLUSTER  (§i)
# ────────────────────────────────────────────────────────────────

cluster_class_table <- table(Cluster = data$cluster, Class = data$class)

cluster_class_table

barplot(cluster_class_table,
        beside = TRUE,
        col = rainbow(4),
        legend = TRUE,
        main = "Class Distribution Across Clusters",
        xlab = "Performance Class",
        ylab = "Count")


# ────────────────────────────────────────────────────────────────
# 6.10  GENDER DISTRIBUTION PER CLUSTER  (§i)
# ────────────────────────────────────────────────────────────────

cluster_gender_table <- table(Cluster = data$cluster, Gender = data$gender)

cluster_gender_table

barplot(cluster_gender_table,
        beside = TRUE,
        col = c("lightblue", "pink"),
        legend = TRUE,
        main = "Gender Distribution Across Clusters",
        xlab = "Gender",
        ylab = "Count")


# ────────────────────────────────────────────────────────────────
# 6.11  OVERALL PERFORMANCE SUMMARY PER CLUSTER  (§i)
# ────────────────────────────────────────────────────────────────

aggregate(data[, c("age", "height_cm", "weight_kg",
                   "gripForce", "sit.ups.counts", "broad.jump_cm", "body.fat_.")],
          by = list(Cluster = data$cluster),
          FUN = mean)


# ────────────────────────────────────────────────────────────────
# 6.12  CLUSTER QUALITY — WITHIN vs BETWEEN SS  (§i)
# ────────────────────────────────────────────────────────────────

cat("Total Within-Cluster SS:", kc$tot.withinss, "\n")
cat("Between-Cluster SS:     ", kc$betweenss, "\n")
cat("Total SS:               ", kc$totss, "\n")
cat("Explained Variance (%)  ", round((kc$betweenss / kc$totss) * 100, 2), "%\n")


# ────────────────────────────────────────────────────────────────
# 6.13  SAVE FINAL DATASET WITH CLUSTER LABELS
# ────────────────────────────────────────────────────────────────

write.csv(data,
          "final_bodyPerformance_with_clusters.csv",
          row.names = FALSE)

# ── END OF PART 6 ───────────────────────────────────────────────
