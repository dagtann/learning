rm(list = ls())
library("rethinking")
data(UCBadmit)
d <- UCBadmit

# Berkely Admissions Model
# A_i ~ Binomial(N_i, p_i)
# logit(p_i) = \alpha_{GID[i]}
# \alpha_j ~ N(0, 1.5)

d <- within(d, {
    gid <- ifelse(applicant.gender == "male", 1, 2)
    }
)
m11.7 <- map2stan(
    alist(
        admit ~ dbinom(applications, p),
        logit(p) <- a[gid],
        a[gid] ~ dnorm(0, 1.5)
    ),
    data = d
)
precis(m11.7, depth=2)

post <- extract.samples(m11.7)
abs_diff <- post[["a"]][, 1] - post[["a"]][, 2]
# relative penguin // difference on log odds scale
rel_diff <- inv_logit(post[["a"]][, 1]) - inv_logit(post[["a"]][, 2])
# absolute shark // difference on probability scale
lapply(list(abs_diff, rel_diff), precis)

postcheck(m11.7)
d[, "dept_id"] <- rep(seq(6), each = 2)
for (i in seq(6)) {
    x <- 1 + 2 * (i - 1)
    y1 <- d$admit[x] / d$applications[x]
    y2 <- d$admit[x + 1] / d$applications[x + 1]
    lines(c(x, x + 1), c(y1, y2), col = rangi2, lwd = 2)
    text(x + 0.5, (y1 + y2) / 2 + .05, d$dept[x], cex = .8, col = rangi2)
}

# Revised Berkely Admissions Model
# A_i ~  Binomial(N_i, p_i)
# logit(p_i) = \alpha_{GID[i]} + \delta_{Dept[i]}
# \alpha_j ~ Normal(0, 1.5)
# \delta_k ~ Normal(0, 1.5)

d <- within(d, did <- as.numeric(dept))
m11.8 <- map2stan(
    alist(
        admit ~ dbinom(applications, p),
        logit(p) <- a[gid] + b[did],
        a[gid] ~ dnorm(0, 1.5),
        b[did] ~ dnorm(0, 1.5)
    ),
    data = d
)
precis(m11.8, depth = 2)
post <- extract.samples(m11.8)
abs_diff <- post[["a"]][, 1] - post[["a"]][, 2]
rel_diff <- inv_logit(post[["a"]][, 1]) - inv_logit(post[["a"]][, 2])
lapply(list(abs = abs_diff, rel = rel_diff), precis)
postcheck(m11.8)
d[, "dept_id"] <- rep(seq(6), each = 2)
for (i in seq(6)) {
    x <- 1 + 2 * (i - 1)
    y1 <- d$admit[x] / d$applications[x]
    y2 <- d$admit[x + 1] / d$applications[x + 1]
    lines(c(x, x + 1), c(y1, y2), col = rangi2, lwd = 2)
    text(x + 0.5, (y1 + y2) / 2 + .05, d$dept[x], cex = .8, col = rangi2)
}



N <- 1000
p <- seq(1, 0, length.out = 1000)
mu <- N * p
sigma <- N * p * (1 - p)
plot(p, mu - sigma, type = 'l', col = "blue")
lines(p, sigma, col = "red")


library("rethinking")
# True DGP
num_days <- 30
y <- rpois(num_days, 1.5)

num_weeks <- 4
y_new <- rpois(num_weeks, .5 * 7)

y_all <- c(y, y_new)
exposure <- c(rep(1, 30), rep(7, 4))
monastary <- c(rep(0, 30), rep(1, 4))
d <- data.frame(y = y_all, days = exposure, monastary = monastary)

# calculate offset
d[, "ldays"] <- log(d$days)

# fit model
fit <- m11.12 <- map(
    alist(
        y ~ dpois(lambda),
        log(lambda) <- ldays + a + b * monastary,
        a ~ dnorm(0, 1),
        b ~ dnorm(0, 1)
    ),
    data = d
)
precis(fit, depth = 2)

post <- extract.samples(fit)
lambda_old <- exp(post$a)
lambda_new <- exp(post$a + post$b)
precis(data.frame(lambda_old, lambda_new))

# Reformulate multinomial regression problem as poisson
data(UCBadmit)
d <- UCBadmit

# logit
fit_logit <- map(
    alist(
        admit ~ dbinom(applications, p),
        logit(p) <-  a,
        a ~ dnorm(0, 10)
    ),
    data = d
)
summary(fit_logit)

# poisson
d[, "rej"] <- d$reject
m_pois <- map2stan(
    alist(
        admit ~ dpois(lambda1),
        rej ~ dpois(lambda2),
        log(lambda1) <- a1,
        log(lambda2) <- a2,
        c(a1, a2) ~ dnorm(0, 100)
    ),
    data = d
)
logistic(coef(fit_logit))
k <- as.numeric(coef(m_pois))
exp(k[1]) / (exp(k[1]) + exp(k[2]))


# Survival models
N <- 2
x <- replicate(1e5, min(runif(N, 1, 100)))
plot(density(x), ylim = c(0, .06))

N2 <- 5
x2 <- replicate(1e5, min(runif(N2, 1, 100)))
lines(density(x2), col = "red", add = TRUE)
