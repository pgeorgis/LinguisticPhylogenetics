#!/usr/bin/env Rscript
source("phyloLing/utils/r/dependencies.R")

# Install 'TreeDist' and 'ape' if not already installed
install_packages_if_needed(c(
	"TreeDist",
	"ape"
))

# Error handling
tryCatch({
    # Read the arguments from the command line
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) != 2) {
        stop("Please provide exactly two Newick tree strings.")
    }

    # Parse the Newick tree strings using 'ape'
    tree1 <- read.tree(text = args[1])
    tree2 <- read.tree(text = args[2])

    # Calculate TreeDist
    result <- TreeDistance(tree1, tree2)

    # Output the result
    cat("TreeDistance: ", result, "\n")

}, error = function(e) {
    cat("Error: ", e$message, "\n")
    quit(status = 1)  # Return an error status code
})
