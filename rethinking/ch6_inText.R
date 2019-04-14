rm(list = ls())
library("rethinking")

# Collider bias simulation
N <- 200
trust <- rnorm(N); news <- rnorm(N); score <- trust + news
selected <- score >= quantile(score, probs = .9)
plot(lm(trust[selected] ~ news[selected]))

# Multicollinearity
data(milk)
d <- apply(milk[, c("kcal.per.g", "perc.fat", "perc.lactose")], 2, scale)
colnames(d) <- c("K", "F", "L")
K <- ncol(d); N <- nrow(d)

stan_model <- "
data {
    int<lower = 0> N;
    int<lower = 0> K;
    matrix[N, K] X;
    vector[N] y;
}
parameters {
    vector[K] beta;
    real<lower = 0> sigma;
}
model {
    y ~ normal(X * beta, sigma);
    beta ~ normal(0, 1);
    sigma ~ exponential(1);
}
"

fit <- stan(model_code = stan_model, data = list(y = d[, 1], X = cbind(1, d[, 2:ncol(d)]), K = K, N = N))
fit
post <- extract(fit)
pairs(post[["beta"]])
# happens because:
pairs(d); cor(d)