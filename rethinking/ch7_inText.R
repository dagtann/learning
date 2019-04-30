rm(list = ls())
packs <- c("rethinking", "tidyverse")
lapply(packs, library, character.only = TRUE)

## 7.1 The problem with parameters ============================================
species <- tibble(
    species = c("afarensis", "africanus", "havilis", "boisei", "rudlfensis",
        "ergaster", "sapiens"),
    brain = c(438, 452, 612, 521, 752, 871, 1350),
    mass = c(37, 35.5, 34.5, 41.5, 55.5, 61, 53.5)

) %>%
    mutate(
        mass_std = (mass - mean(mass)) / sd(mass),
        brain_std = brain / max(brain)
    )

code_m7.1 <- "
    data {
        int<lower = 0> N;
        int<lower = 1> K;
        matrix[N, K] X;
        vector[N] y;
    }
    parameters {
        // real<lower = 0> log_sigma;
        vector[K] beta;
    }
    model {
        y ~ normal(X * beta, 0.001); // exp(log_sigma));
        beta[1] ~ normal(0.5, 1);
        beta[2:K] ~ normal(0, 10);
        // log_sigma ~ normal(0, 1);
    }
"

N <- nrow(species)
X <- cbind(1, species[["mass_std"]]); K <- ncol(X)
y <- species[["brain_std"]]

m7.1 <- rethinking::map(
    alist(
        brain_std ~ dnorm( mu , exp(log_sigma) ),
        mu <- a + b*mass_std,
        a ~ dnorm( 0.5 , 1 ),
        b ~ dnorm( 0 , 10 ),
        log_sigma ~ dnorm( 0 , 1 ) ), data=as.data.frame(species)
    )
)
m7.1_stan <- stan(
    model_code = code_m7.1, data = list(N = N, K = K, X = X, y = y), init = 0
)

### Demonstration R2
post <- rstan::extract(m7.1_stan)
yhat <- tcrossprod(X, post$beta)
e <- apply(yhat, 1, mean) - y
1 - var2(e) / var2(y)

calc_R2 <- function(stan_model, X, y){
    yhat <- tcrossprod(X, rstan::extract(stan_model)[["beta"]])
    e <- apply(yhat, 1, mean) - y
    1 - rethinking::var2(e) / rethinking::var2(y)
}

### Demonstrate overfitting
generate_polynomials <- function(x, max_degree){
    vapply(seq(max.degree), FUN = function(d){x ^ d}, numeric(length(x)))
}
max_degree <- 6
R2_estimates <- vector("numeric", length = max_degree)
X_pred <- cbind(1,
    seq(min(species[["mass_std"]]), max(species[["mass_std"]]),
        length.out = 100
    )
)
par(mfrow = c(2, 3))
for(i in seq(max_degree)){
    X <- cbind(1, generate_polynomials(species[["mass_std"]], i))
    tmp <- stan(
        model_code = code_m7.1, data = list(N = N, K = ncol(X), X = X, y = y),
        init = 0
    )
    R2_estimates[i] <- calc_R2(tmp, X, y)
    yhat <- tcrossprod(
        cbind(1, generate_polynomials(X_pred[, 2], i)),
        rstan::extract(tmp)[["beta"]]
    )
    ci <- apply(yhat, 1, PI)
    yhat <- apply(yhat, 1, mean)
    plot(species[["mass_std"]], y)
    lines(X_pred[, 2], yhat)
    shade(ci, X_pred[, 2])
    text(x = min(species[["mass_std"]]) + .75, y = max(species[["brain_std"]]) - .01,
        labels = round(R2_estimates[i], 2)
    )
}
### Notice: Chapter results cannot be reproduced with STAN unless sigma is always
### set to .001

### 7.1.2 Underfitting
X <- matrix(1, nrow = nrow(species))
tmp <- stan(
    model_code = code_m7.1, data = list(N = N, K = ncol(X), X = X, y = y),
    init = 0
)

# sensitivity to data
X <- cbind(1, generate_polynomials(species[["mass_std"]], 6))
dev.off()
plot(species[["mass_std"]], y, col = rangi2, pch = 19)
for (i in seq(nrow(species))) {
    tmp <- stan(
        model_code = code_m7.1,
        data = list(N = nrow(X) - 1, K = ncol(X), X = X[-i, ], y = y[-i]),
        init = 0
    )
    yhat <- tcrossprod(
        cbind(1, generate_polynomials(X_pred[, 2], i)),
        rstan::extract(tmp)[["beta"]]
    )
    yhat <- apply(yhat, 1, mean)
    lines(X_pred[, 2], yhat)
}

# 7.2 Entropy and accuracy
# 7.2.2 Information and uncertainty
# Information: The reduction in uncertainty derived from learning an outcome.
#       1. The measure should be continuous.
#       2. The measure should increase in the number of possible events.
#       3. The measure of uncertainty should be additive.
# Formula: H(p) = -E(log(pi)) = - \sum_{i=1}^n p_i log(p_i)
#       - sum(p * ifelse(p == 0, 0, log(p))

# 7.2.3 From Entropy to accuracy
# H quantifies uncertainty. Now, how do we use it to measure the distance from
# one probability distribution to another.
# -> Divergence: The additional uncertainty induced by using probabilities from
#   one distribution to describe another distribution.
# Kullback-Leibler divergence
#   D_{KL}(p. q) = \sum_{i = 1}^N p_i(log(p_i) - log(q_i)) = \sum_{i = 1}^N log(\frac{p_i}{q_i})
# -> if p = q then D_{KL} = 0
# Key step in derivation: Cross entropy
# H(p, q) = - \sum_{i=1}^N p_i log(q_i)
# Divergence measures the *additional* entropy induced by using q to describe p
# So divergence really is measuring how far q is from the target p, in units of entropy.
# D_{KL}(p. q) = H(p, q) - H(p) = \sum_{i = 1}^N p_i(log(p_i) - log(q_i)) - (-E(log(pi))) = \sum_{i = 1}^N log(\frac{p_i}{q_i})
steps <- 100
pdta <- expand.grid(p = .3, q = seq(0.01, .99, length.out = steps))
pdta[, "divergence"] <- pdta[, 1] * log(pdta[, 1] / pdta[, 2])
pdta[, "divergence"] <- pdta[, "divergence"] + log((1- pdta[, 1]) / (1- pdta[, 2]))
plot(divergence ~ q, data = pdta, type = "l", col = rangi2, ylab = "Divergence of q from p")
abline(v = .3, lty = "dashed")
text(x = .35, y = 1.5, labels = expression(q == p))

# Problem: How do we estimate divergence in real statistical modeling
# -> Answer: Deviance
# Complicating circumstance: p is unknownn, i.e. the target probability
# distribution is unknown
# Note: We are only interested in comparing the divergence of different candidate
# descriptions of p, say q and r. E(log(p_i)) is inconsequential for the distance
# between these candidate descriptions. The only requirement of that comparison
# is each model's average log-probability E(log(q_i)) & E(log(p_i)). These can
# be approximated by the sum of logged probabilities: S(q) = \sum_{i = 1}^N log(q_i)
# Bayesian version: log-probabilty predictive density (lppd)
# lppd(y, \Theta) = \sum_{i = 1}^N log(\frac{1}{S} \sum_{s = 1}^S p(y_i | \Theta_s))

set.seed(1) lppd( m7.1 , n=1e4 )

calculate_lppd <- function(stan_model){

}
str(post)


































