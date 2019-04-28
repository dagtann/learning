rm(list = ls())

# Example: Attribute selection with entropy ===================================
entropy <- function(x, base = 2) { # calculate entropy
    X <- outer(x, unique(x), "==")
    X <- apply(X, 2, mean)
    as.numeric(-log(X, base = base) %*% X)
}


information_gain <- function(parent, child, score = FALSE){
    child_labels <- unique(child)
    K <- length(child_labels)
    child_per <- apply(outer(child, child_labels, "=="), 2, mean)
    gain <- vector("numeric", K)
    for (i in seq(K)) { gain[i] <- entropy(parent[child == child_labels[i]]) }
    if (score) {
        return(as.numeric(entropy(parent) - child_per %*% gain))
    }
    cbind.data.frame(label = child_labels, per = child_per, entropy = gain)
}


# Prep data
mushrooms <- read.table(
    "/Users/dag/github/learning/ds4Business/agaricus-lepiota.data",
    header = FALSE, sep = ",",
    col.names = c("edible", "cap_shape", "cap_surface", "cap_color", "bruises?",
        "odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color",
        "stalk_shape", "stalk_root", "stalk_surface_above_ring",
        "stalk_surface_below_ring", "stalk_color_above_ring",
        "stalk_color_below_ring", "veil_type", "veil_color", "ring_number",
        "ring_type", "spore_print_color", "population", "habitat"
    ),
    stringsAsFactors = FALSE
)
set.seed(01012016)
train <- sample(seq(nrow(mushrooms)), 5644, replace = FALSE) # N_obs stated
mushrooms_train <- mushrooms[train, ]

# Test run feature selection
entropy(mushrooms_train$edible) # in text: .96 (p. 59)
pdta <- information_gain(mushrooms_train$edible, mushrooms_train$odor)
ggplot( # entropy diagram
    data = pdta, aes(x = reorder(label, entropy), y = entropy, fill = per)
) +
    geom_bar(stat = "identity")
pdta[, 2] %*% pdta[, 3] # entropy in edible left after knowing odor
entropy(mushrooms_train$edible) - pdta[, 2] %*% pdta[, 3] # information gain

# Calculate information gain for each candidate feature.
igs <- apply(mushrooms_train[, -1], 2,
    function(x){
        information_gain(parent = mushrooms_train$edible, child = x, score = TRUE)
    }
)
sort(igs)
## END