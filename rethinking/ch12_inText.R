library("rethinking")

## Over-dispersed outcomes ====================================================
# Observed variance exceeds the analytical expectation after conditioning on all
# predictors
# can confound, hide effects or cause spurious inferences
# How to deal w/i:

# (1) Beta-Binomial Example
plot_beta <- function(prob = .5, theta = 2){
    a <- prob * theta
    b <- (1 - prob) * theta
    curve(dbeta(x, shape1 = a, shape2 = b), from = 0, to = 1)
}
plot_beta(p = .5, theta = .001)
plot_beta(p = .5, theta = 10)

data(UCBadmit)
d <- UCBadmit
d <- within(d, {
    gid <- ifelse(applicant.gender == "male", 1L, 2L)
})
fit <- map2stan(
    alist(
        admit ~ dbetabinom(applications, pbar, theta),
        logit(pbar) <- a[gid],
        a[gid] ~dnorm(0, 1.5),
        theta ~ dexp(1)
    ),
    data = d
)
post <- extract.samples(fit)
post$da <- post$a[, 1] - post$a[, 2]
lapply(post, summary)

# draw the implied beta distribution
gid <- 2
curve(
    dbeta2(x, mean(logistic(post$a[, gid])), mean(post$theta)),
    from = 0, to = 1,
    ylab = "density", xlab = "probability admit", ylim = c(0, 3), lwd = 2
)
# draw 50 beta distributions sampled from the posterior
for (i in seq(50)){
    p <- logistic(post$a[i, gid])
    theta <- post$theta[i]
    curve(dbeta2(x, p, theta), add = TRUE, col = col.alpha("black", .2))
}
postcheck(fit)

# (2) Gamma Possion Example (Negative-binomial)
data(Kline)
d <- Kline
d$P <- scale(log(d$population))
d$contact_id <- ifelse(d$contact == "high", 2L, 1L)

dat2 <- list(
    T = d$total_tools,
    P = d$population,
    cid = d$contact_id
)

m12.3 <- map2stan(
    alist(
        T ~ dgampois(lambda, phi),
        lambda <- exp(a[cid]) * P ^ b[cid] / g,
        a[cid] ~ dnorm(1, 1),
        b[cid] ~ dexp(1),
        g ~ dexp(1),
        phi ~ dexp(1)
    ), data = dat2, chains = 4, iter = 8000, cores = 3
)
precis(m12.3, depth = 2)

## Zero-inflated outcomes
prob_drink <- .2 # monks drink 20% of the time
rate_work <- 1 # when not drinking, monks finish one book per day on average
N <- 365 # sim 1 year of exposure

set.seed(365)
drink <- rbinom(N, 1, prob_drink)
y <- (1 - drink) * rpois(N, rate_work)
simplehist(y)
zeros_drink <- sum(drink)
zeros_work <- sum(drink == 0 & y == 0)
zeros <- sum(y == 0)
zeros_drink; zeros_work; zeros
lines(c(0,0), c(zeros_work, zeros), lwd = 2, col = rangi2)

# 0s are inflated relative to a standard poisson process b/c there is
# heterogeneity in the process that generates 0s
# (1-p) -> observe y > 0 OR y = 0
# ^
# |
# O -> p -> observe y = 0

m12.4 <- map2stan(
    alist(
        y ~ dzipois(p, lambda),
        logit(p) <- ap,
        log(lambda) <- al,
        ap ~ dnorm(-1.5, 1),
        al ~ dnorm(1, 0.5)
    ),
    data = list(y = as.integer(y)), chains = 4
)
precis(m12.4)
inv_logit(-1.29) # estimated probability of drinking
exp(.01) # mean estimated rate of works finished

## 12.3 Ordered Categorical Outcomes ------------------------------------------
data(Trolley)
d <- Trolley

summary(d)
simplehist(d$response, xlim = c(1, 7), xlab = "response")

# convert response categories to log-cumulative-odds
pr_k <- prop.table(table(d$response))
cum_pr_k <- cumsum(pr_k)

plot(seq(cum_pr_k), cum_pr_k, type = "b", xlab = "response",
     ylab = "Cumulative Probability", ylim = c(0, 1))

calc_cum_odds <- function(pr, log = TRUE){
    out <- vector("numeric", length(pr))
    for (i in seq(pr)) {
        out[i] <- sum(pr[1:i]) / (1 - sum(pr[1:i]))
    }
    if(log) out <- log(out)
    names(out) <- seq(length(pr))
    return(out)
}
calc_cum_odds(pr_k)

# basic intercept only model
m12.5 <- map2stan( # won't work
    alist(
        R ~ dordlogit(0, cutpoints),
        cutpoints ~ dnorm(0, 1.5)
    ),
    data = list(R = d$response), chains = 4, cores = 3
)

