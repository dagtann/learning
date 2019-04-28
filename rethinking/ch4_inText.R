library("rethinking")
data(Howell1)
str(Howell1)
precis(Howell1)

d2 <- Howell1[Howell1[["age"]] >= 18, ]

# plot the prior N(178, 20)
curve(dnorm(x, 178, 20), 0, 400)
# -> the **average** height is between 140 and 220

# prior predictive simulation
# sample from the prior to perform sanity checks on the **average** height
N <- 10e4
mu_sample <- rnorm(N, 178, 20)
sigma_sample <- runif(N, 0, 50)
implied_height <- rnorm(N, mu_sample, sigma_sample)
dens(implied_height); rug(implied_height)
summary(implied_height) # freak height below 0
mean(implied_height < 0)
mean(implied_height > 272) # roughly 1 % taller than tallest known man

flist <- alist(
    height ~ dnorm(mu, sigma) ,
    mu ~ dnorm(178, 20) ,
    sigma ~ dunif(0, 50)
)
start <- list(mu = mean(d2$height), sigma = sd(d2$height))
m4.1 <- map(flist, data = d2)


model <- "
    data {
        int<lower = 0> N ;
        vector[N] height ;
    }
    parameters {
        real<lower = 0> mu ;
        real<lower = 0, upper=50> sigma ;
    }
    model {
        height ~ normal(mu, sigma) ;
        mu ~ normal(178, 20) ;
        sigma ~ uniform(0, 50) ;
    }

"
m4.1_stan <- stan(model_code = model, data = list(N = nrow(d), height = d$height))


list_of_draws <- extract(fit)
matrix_of_draws <- as.matrix(fit)
dim(matrix_of_draws); head(matrix_of_draws)

## Linear regression strategy
plot(d2$height ~ d2$weight)

### prior predictive simulation
set.seed(2971)
N_replications <- 100
beta <- cbind(
    rnorm(N_replications, 178, 20),
    rlnorm(N_replications, 0, 1)
    # try rnorm(N_replications, 0, 10) for non-sensible results
    # gist:
    #   (a) non-sensible predictions, e.g. < 0, > 272
    #   (b) for the average human population, height should not decline in weight
    # -> log b for all positiive slopes & saner predictions
)
plot(NULL, xlim = range(d2$weight), ylim = c(-100, 400))
for (i in 1:N_replications){
    curve(
        cbind(1, x - mean(d2$weight)) %*% beta[i, ],
        from = min(d2$weight), to = max(d2$height), add = TRUE
    )
}
abline(h = c(0, 272))


model2 <- "
    data {
        int<lower = 0> N ;
        int<lower = 0> K;
        vector[N] height ;
        matrix[N, K] X ;
    }
    parameters {
        vector[K] beta;
        real<lower = 0, upper = 50> sigma ;
    }
    model {
        height ~ normal(X * beta, sigma) ;
        beta[1] ~ normal(178, 20) ;
        for (k in 2:K) {
            beta[k] ~ normal(0, 10) ;
        }
        sigma ~ uniform(0, 50) ;
    }

"

X <- cbind(1, scale(d2$weight, scale = FALSE))
N <- nrow(X)
K <- ncol(X)
y <- d2$height
m4.1_stan <- stan(model_code = model2, data = list(N = nrow(X), K = ncol(X), X = X, height = d2$height))
(m4.1_stan)
lm(y ~ X[, -1]) # sanity check

post <- extract(m4.1_stan)
e_beta <- apply(post[["beta"]], 2, mean)
plot(d2$weight, d2$height)
mu <- mean(d2$weight)
x <- scale(d2$weight, scale = FALSE)
x <- seq(min(x), max(x))
yhat <- cbind(1, x) %*% e_beta
lines( # add mean effect to plot
    x + mu, yhat
)
# add model_uncertainty
plot(density(cbind(1, 50 - mu) %*% t(post[["beta"]]))) # example at 50 kg
yhat_model_uncertainty <- cbind(1, x) %*% t(post[["beta"]])
plot(range(x + mu), range(d2$height), type = "n")
for (i in seq(100)) { points(x + mu, yhat_model_uncertainty[, i])}

## summarize posterior point predictions
mu_seq <- apply(yhat_model_uncertainty, 1,  mean)
lines(x + mu, mu_seq, col = "red")
mu_hpdi <- apply(yhat_model_uncertainty, 1,  HPDI, prob = .89)
shade(mu_hpdi, x + mu)

# create prediction intervals
e_sim <- matrix(FALSE, nrow = nrow(yhat_model_uncertainty), ncol = ncol(yhat_model_uncertainty))
for(j in ncol(e_sim)){
    e_sim[, j] <- rnorm(nrow(e_sim), 0, post[["sigma"]][j])
}
height_sim <- yhat_model_uncertainty + e_sim
height_PI <- apply(height_sim, 2, PI, prob = .89)
shade(height_PI, x + mu)


## Polynomial regression
X <- cbind(1, scale(Howell1$weight), scale(Howell1$weight) ^ 2, scale(Howell1$weight) ^ 3)
N <- nrow(X); K <- ncol(X)
m4.5_stan <- stan(model_code = model2, data = list(N = N, K = K, X = X, height = Howell1$height), iter = 10e3)
post <- extract(m4.5_stan)
yhat <- X %*% t(post[["beta"]])
plot(X[, 2], apply(yhat, 1, mean)) # sanity check: Does the output match the book?

## Splines
library(rethinking); library(tidyverse); library(rstan)
cherries <- read.csv2("./cherry_blossoms.csv")
str(cherries)
levels(cherries$temp)
cherries[, "temp"] <- as.numeric(as.character(cherries[, "temp"]))
summary(cherries)
cherries <- cherries[ complete.cases(cherries), ]
n_knots <- 15
knots <- quantile(cherries$year, probs = seq(0, 1, length.out = n_knots))
B <- splines::bs(cherries$year, knots = knots[-c(1, n_knots)], degree = 3, intercept = TRUE)

X <- cbind(1, B)
y <- cherries$temp
N <- length(y); K <- ncol(X)
model <- "
    data {
        int<lower = 0> N;
        int<lower = 0> K;
        matrix[N, K] X;
        vector[N] y;
    }
    parameters {
        real<lower = 0> sigma;
        vector[K] beta;
    }
    model {
        y ~ normal(X * beta, sigma);
        beta[1] ~ normal(6, 10);
        for (k in 2:K) {
            beta[k] ~ normal(0, 1);
        }
        sigma ~ exponential(1);
    }

"
m4.7 <- map(
    alist(
        T ~ dnorm(mu, sigma),
        mu <- a + B %*% w,
        a ~ dnorm(6, 10),
        w ~ dnorm(0, 1),
        sigma ~ dexp(1)
    ),
    data = list(T = cherries$temp, B = B),
    start = list(w = rep(0, ncol(B)))
)
m4.6_stan <- stan(
    model_code = model,
    data = list(N = N, K = K, X = X, y = y),
    cores = parallel::detectCores() - 1
)
post <- rstan::extract(m4.6_stan)
yhat <- X %*% t(post[["beta"]])

plot(cherries$year, cherries$temp)
lines(cherries$year, apply(yhat, 1, mean), col = "red")
shade(apply(yhat, 1, HPDI, prob = .89), cherries$year)