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

## add a predictor
plot(d$height ~ d$weight)
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
        for (k in 1:K) {
            beta[k] ~ normal(0, 10) ;
        }
        sigma ~ uniform(0, 50) ;
    }

"

X <- cbind(1, scale(d2$weight))
N <- nrow(X)
K <- ncol(X)
y <- d2$height
m4.1_stan <- stan(model_code = model2, data = list(N = nrow(d3), K = ncol(X), X = X, height = d2$height))
(m4.1_stan)
solve(t(X) %*% X) %*% t(X) %*% y # sanity check