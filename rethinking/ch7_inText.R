rm(list = ls())
packs <- c("rethinking", "tidyverse")
lapply(packs, library, character.only = TRUE)

species <- tibble(
    species = c("afarensis", "africanus", "havilis", "boisei", "rudlfensis",
        "ergaster", "sapiens"),
    brain = c(438, 452, 612, 521, 752, 871, 1350),
    mass = c(37, 35.5, 34.5, 41.5, 55.5, 61, 53.5)

) %>%
    mutate(mass_std = (mass - mean(mass)) / sd(mass), brain_std = brain / max(brain))

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

m7.1_stan <- stan(model_code = code_m7.1, data = list(N = N, K = K, X = X, y = y), init = 0)
post <- rstan::extract(m7.1)
yhat <- tcrossprod(X, post$beta)
e <- apply(yhat, 1, mean) - y
1 - var2(e) / var2(y)

calc_R2 <- function(stan_model, X, y){
    yhat <- tcrossprod(X, rstan::extract(stan_model)[["beta"]])
    e <- apply(yhat, 1, mean) - y
    1 - rethinking::var2(e) / rethinking::var2(y)
}
generate_polynomials <- function(x, max.degree = 2){
    vapply(seq(max.degree), FUN = function(d){x ^ d}, numeric(length(x)))
}

max_degree <- 6
R2_estimates <- vector("numeric", length = max_degree)
X_pred <- cbind(1, seq(min(species[["mass_std"]]), max(species[["mass_std"]]), length.out = 100))
par(mfrow = c(2, 3))
for(i in seq(max_degree)){ 
    X <- cbind(1, generate_polynomials(species[["mass_std"]], i))
    tmp <- stan(model_code = code_m7.1, data = list(N = N, K = ncol(X), X = X, y = y), init = 0)
    R2_estimates[i] <- calc_R2(tmp, X, y)
    yhat <- cbind(1, generate_polynomials(X_pred[, 2], i)) %*% t(rstan::extract(tmp)[["beta"]])
    yhat <- apply(yhat, 1, mean)
    plot(species[["mass_std"]], y)
    lines(X_pred[, 2], yhat)
}
R2_estimates


m7.1 <- rethinking::map(
    alist(
        brain_std ~ dnorm( mu , exp(log_sigma) ),
        mu <- a + b*mass_std, a ~ dnorm( 0.5 , 1 ),
        b ~ dnorm( 0 , 10 ),
        log_sigma ~ dnorm( 0 , 1 ) 
    ),
    data = as.data.frame(species)
)
m7.2 <- rethinking::map2stan(
    alist(
        brain_std ~ dnorm(mu , exp(log_sigma)),
        mu <- a + b[1]*mass_std + b[2]*mass_std^2,
        a ~ dnorm( 0.5 , 1 ),
        b ~ dnorm( 0 , 10 ),
        log_sigma ~ dnorm( 0 , 1 )
    ),
    data = as.data.frame(species) , start=list(b=rep(0,2)) 
)

m7.6 <- rethinking::map2stan(
    alist(
        brain_std ~ dnorm( mu , 0.001 ),
        mu <- a + b[1]*mass_std + b[2]*mass_std^2 + b[3]*mass_std^3 +
            b[4]*mass_std^4 + b[5]*mass_std^5 + b[6]*mass_std^6,
        a ~ dnorm( 0.5 , 1 ),
        b ~ dnorm( 0 , 10 )
    ),
    data = as.data.frame(species),
    start = list(b=rep(0,6))
)
par(mfrow = c(1, 1))

d <- as.data.frame(species)
post <- extract.samples(m7.1_stan)
mass_seq <- seq( from=min(d$mass_std) , to=max(d$mass_std) , length.out=100 )
l <- link( m7.6 , data=list( mass_std=mass_seq ) )
mu <- apply( l , 2 , mean ) 
ci <- apply( l , 2 , PI ) 
plot( brain_std ~ mass_std , data=d ) 
lines( mass_seq , mu ) 
shade( ci , mass_seq )

cat(m7.6@model)



summary(lm(y ~ X[, -1]))