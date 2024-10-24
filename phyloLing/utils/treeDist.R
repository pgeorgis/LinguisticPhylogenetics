#!/usr/bin/env Rscript

# Function to check if a package is installed and install it if necessary
install_if_needed <- function(package_name) {
    if (!require(package_name, quietly = TRUE, character.only = TRUE)) {
        install.packages(package_name, repos = "https://cran.r-project.org")
        suppressPackageStartupMessages(library(package_name, character.only = TRUE))
    } else {
        suppressPackageStartupMessages(library(package_name, character.only = TRUE))
    }
}

# Install 'TreeDist' and 'ape' if not already installed
install_if_needed("TreeDist")
install_if_needed("ape")

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
