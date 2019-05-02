rm(list = ls())
library(e1071)

# SUPPORT VECTOR CLASSIFIER
# Generate non-linearly seperable data
set.seed(1)
n_features <- 2
n_train <- 20
X <- matrix(rnorm(n_train * n_features), ncol = n_features)
y <- rep(c(-1, 1), each = n_train / 2)
filter <- which(y == 1)
X[filter, ] <- X[filter, ] + 1
rm(filter)
plot(X, col = 3 - y, pch = 20)

# fit SVM
# note: y must be factor
dta <- data.frame(X, y = factor(y))
fit_svm <- svm(y ~ ., data = dta, kernel = "linear", cost = 10, scale = FALSE)
summary(fit_svm)
plot(fit_svm, dta)

# identify support vectors
fit_svm$index

# tune Cost parameter
set.seed(1)
fit_tuned <- tune(
    svm, y ~ ., data = dta, kernel = "linear", scale = FALSE,
    ranges = list(cost = c(.001, .01, .1, 1, 5, 10, 100))
)
summary(fit_tuned)
fit_best <- fit_tuned$best.model
summary(fit_best)

# predict test data
xtest <- matrix(rnorm(n_train * n_features), ncol = n_features)
ytest <- sample(c(-1, 1), n_train, replace = TRUE)
filter <- which(ytest == 1)
xtest[filter] <- xtest[filter] + 1
test <- data.frame(xtest, y = as.factor(ytest))

yhat <- predict(fit_svm, newdata = test)
table(yhat, ytest)

# generate linearly separable data
filter <- which(y == 1)
X[filter] <- X[filter] + 1.5
plot(X, col = (y + 5) / 2)
dta <- data.frame(X, y = as.factor(y))
svm_fit <- svm(y ~ ., data = dta, kernel = "linear", cost = 1)
summary(svm_fit)
plot(svm_fit, dta)

# SUPPORT VECTOR MACHINE
set.seed(1)
n_train <- 200
X <- matrix(rnorm(n_train * n_features), ncol = n_features)
X[seq(100), ] <- X[seq(100), ] + 2
X[101:150, ] <- X[101:150, ] - 2
y <- c(rep(1, 150), rep(2, 50))
dta <- data.frame(X, y = factor(y)); rm(X, y)
plot(dta[, -3], col = dta[, 3])


train <- sample(200, 100)
train_dta <- dta[train, ]
svmfit <- svm(y ~ ., data = train_dta, kernel = "radial", gamma = 1, cost = 1)
plot(svmfit, dta)
summary(svmfit)
svmfit <- tune(
    svm, y ~ ., data = train_dta, kernel = "radial",
    ranges = list(
        cost = c(1 * 10 ^ (-1:3)),
        gamma = c(.5, 1:4)
    )
)
summary(svmfit)






















