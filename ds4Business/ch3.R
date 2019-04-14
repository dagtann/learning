rm(list = ls())

# Extract informative features
entropy <- function(x, base = 2) { # calculate entropy
    X <- outer(x, unique(x), "==")
    X <- apply(X, 2, mean)
    as.numeric(-log(X, base = base) %*% X)
}
information_gain <- function(parent, child){
    entropy_c <- vapply(unique(child),
        function(x) {
            mean(child == x) * entropy(parent[child == x])
        },
        numeric(1)
    )
    entropy(parent) - sum(entropy_c)
}
mushrooms <- read.table(
    "/Users/dag/github/learning/ds4Business/agaricus-lepiota.data",
    header = FALSE, sep = ',',
    col.names = c("edible", "cap_shape", "cap_surface", "cap_color", "bruises?", 
        "odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color",
        "stalk_shape", "stalk_root", "stalk_surface_above_ring",
        "stalk_surface_below_ring", "stalk_color_above_ring",
        "stalk_color_below_ring", "veil_type", "veil_color", "ring_number",
        "ring_type", "spore_print_color", "population", "habitat"
    )
)
mushrooms[, "propensity"] <- runif(nrow(mushrooms), 0, 1)
train <- sample(seq(nrow(mushrooms)), 5644, replace = FALSE) # sample N_obs stated
mushrooms_train <- mushrooms[train, ]
entropy(mushrooms_train$edible)

parent <- mushrooms_train$edible
child <- mushrooms_train$odor
child_labels <- unclass(unique(child))
child_cats <- apply(outer(child, child_labels, "=="), 2, mean)

entropy(parent[child == x])
sapply(child_labels,
    function(parent = parent, child = child, x) {
        entropy(parent[child == x])
    }
)


names(child_cats) <- child_labels
gain <- vector("numeric", length(child_labels))
names(gain) <- child_labels
for (i in child_labels) {
    gain[i] <- child_cats[i] * entropy(mushrooms[mushrooms$odor == i, "edible"])
}
entropy(mushrooms$edible) - sum(gain)
information_gain(mushrooms$edibly, mushrooms$odor)
vapply(unique(mushrooms$spore.print.color), function(x){

})
information_gain(mushrooms$edible, mushrooms$spore.print.color)