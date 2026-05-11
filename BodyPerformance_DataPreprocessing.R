# Part 1: Preprocessing + EDA + Visualizations

# to visualize the correlation matrix
install.packages("corrplot")

# Load Dataset
data <- read.csv("C:/Users/Menna Thabet/Downloads/bodyPerformance.csv")

# Explore Dataset

head(data)

str(data)

summary(data)

dim(data)

names(data)


# Check Missing Values

sum(is.na(data))

colSums(is.na(data))


# Check Duplicate Rows

sum(duplicated(data))


# Remove Duplicates
data <- unique(data)


# Encoding categorical variables

data$gender <- as.factor(data$gender)

data$class <- as.factor(data$class)

# Check Dataset Structure Again

str(data)

summary(data)



# PREPROCESSING

# Outlier Detection

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



# Remove impossible blood pressure values

data <- subset(data, systolic > 0)

data <- subset(data, diastolic > 0)


# Remove impossible grip force values

data <- subset(data, gripForce > 0)


# Normalize Numerical Data

normalize <- function(x)
{
  return((x - min(x)) / (max(x) - min(x)))
}

data$age_norm <- normalize(data$age)

data$weight_norm <- normalize(data$weight_kg)

data$bodyfat_norm <- normalize(data$body.fat_.)

data$grip_norm <- normalize(data$gripForce)


# EXPLORATORY DATA ANALYSIS (EDA)

# Class Distribution

counts <- table(data$class)

barplot(counts,
        col = rainbow(4),
        main = "Class Distribution",
        xlab = "Class",
        ylab = "Count")


# Gender Distribution

gender_count <- table(data$gender)

pie(gender_count,
    labels = gender_count,
    col = rainbow(length(gender_count)),
    main = "Gender Distribution")


# Age Distribution

hist(data$age,
     col = "lightblue",
     breaks = 10,
     main = "Age Distribution",
     xlab = "Age")


# Weight Distribution

hist(data$weight_kg,
     col = "lightgreen",
     breaks = 10,
     main = "Weight Distribution",
     xlab = "Weight")


# Body Fat Distribution

hist(data$body.fat_.,
     prob = TRUE,
     col = "grey",
     main = "Body Fat Distribution",
     xlab = "Body Fat %")

lines(density(data$body.fat_.),
      col = "blue",
      lwd = 2)


# Grip Force Distribution

hist(data$gripForce,
     col = "orange",
     breaks = 10,
     main = "Grip Force Distribution",
     xlab = "Grip Force")


# Body Fat vs Class

boxplot(body.fat_. ~ class,
        data = data,
        col = rainbow(4),
        main = "Body Fat Percentage by Class",
        xlab = "Class",
        ylab = "Body Fat %")


# Grip Force by Gender

boxplot(gripForce ~ gender,
        data = data,
        col = c("lightblue", "pink"),
        main = "Grip Force by Gender",
        xlab = "Gender",
        ylab = "Grip Force")

# Sit-Ups by Class

boxplot(sit.ups.counts ~ class,
        data = data,
        col = rainbow(4),
        main = "Sit-Ups Counts by Class",
        xlab = "Class",
        ylab = "Sit-Ups Counts")

# Broad Jump by Class

boxplot(broad.jump_cm ~ class,
        data = data,
        col = rainbow(4),
        main = "Broad Jump by Class",
        xlab = "Class",
        ylab = "Broad Jump")

# Age vs Sit-Ups

plot(data$age,
     data$sit.ups.counts,
     col = "blue",
     pch = 19,
     main = "Age vs Sit-Ups Counts",
     xlab = "Age",
     ylab = "Sit-Ups Counts")


# Grip Force vs Broad Jump

plot(data$gripForce,
     data$broad.jump_cm,
     col = "red",
     pch = 19,
     main = "Grip Force vs Broad Jump",
     xlab = "Grip Force",
     ylab = "Broad Jump")

# Height vs Weight

plot(data$height_cm,
     data$weight_kg,
     col = "darkgreen",
     pch = 19,
     main = "Height vs Weight",
     xlab = "Height (cm)",
     ylab = "Weight (kg)")


# Systolic vs Diastolic

plot(data$systolic,
     data$diastolic,
     col = "purple",
     pch = 19,
     main = "Systolic vs Diastolic Pressure",
     xlab = "Systolic",
     ylab = "Diastolic")


# Dot Plot

dotchart(data$gripForce,
         labels = row.names(data),
         cex = 0.5,
         main = "Grip Force Dot Plot",
         xlab = "Grip Force")


# Correlation Analysis

numeric_data <- data[, sapply(data, is.numeric)]

cor_matrix <- cor(numeric_data)

cor_matrix

# Correlation Plot

library(corrplot)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.cex = 0.7)


# Scatter Plot Matrix

dev.off()

plot(numeric_data)

# Save Clean Dataset

write.csv(data,
          "cleaned_bodyPerformance.csv",
          row.names = FALSE)

# END OF Part 1

