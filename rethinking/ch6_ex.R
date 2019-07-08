rm(list = ls())
options(help_type = "html")
library("rethinking")
library("tidyverse")
library("dagitty")

# 6H1 Use the Waffle House data, data(WaffleDivorce), to find the total causal
# influence of number of Waffle Houses on divorce rate. Justify your model or
# models with a causal graph.
data(WaffleDivorce)
help(WaffleDivorce)

# Define causal model
g1 <- dagitty( "dag {
    w <- s -> d
    w <- p -> d
}")
plot(graphLayout(g1))
print(impliedConditionalIndependencies(g1))

# extract data for fit
data_to_fit <- select(WaffleDivorce, WaffleHouses, Population, Divorce, South) %>%
    rename("d" = Divorce, "w" = WaffleHouses, "p" = Population, "s" = South)

# test implications
fit <- lm(Divorce ~ WaffleHouses + Population + South, data = WaffleDivorce)
cor.test(data_to_fit[, c("p")], data_to_fit[, c("s")])
