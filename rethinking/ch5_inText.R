rm(list = ls())
library("rethinking")
data(WaffleDivorce)

X <- cbind(1, scale(WaffleDivorce$MedianAgeMarriage), scale(WaffleDivorce$Marriage))
y <- as.vector(scale(WaffleDivorce$Divorce))
N <- nrow(X); K <- ncol(X)

stan_model <- "
data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    vector[N] y;
}
parameters {
    vector[K] beta;
    real<lower=0> sigma;
}
model {
    y ~ normal(X * beta, sigma);
    for (i in 1:K) {
        beta[K] ~ normal(0, .5);
    }
    sigma ~ exponential(1);
}

"
fit_age <- stan(model_code = stan_model, data = list(N = N, K = K, X = X, y = y))
fit_age
post <- extract(fit_age)
X_pred <- expand.grid(
    1,
    seq(min(X[, 2]), max(X[, 2]), length.out = 30),
    seq(min(X[, 3]), max(X[, 3]), length.out = 30)
)
yhat <- tcrossprod(as.matrix(X_pred), post[["beta"]])
dim(yhat)
mu_hat <- apply(yhat, 1, mean)
sample_fits <- sample(seq(ncol(yhat)), 100, replace = TRUE)
plot(X_pred[, 2], mu_hat, type = "n")
for(i in seq(length(sample_fits))){
    lines(X_pred[, 2], yhat[, i])
}
lines(X_pred[, 2], mu_hat, col = "red")