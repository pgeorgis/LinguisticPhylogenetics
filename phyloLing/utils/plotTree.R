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

install_if_needed("ape")
install_if_needed("phytools")
install_if_needed("stringr")

#Function for modifying format of tree tip labels (doculect names) 
#in order to match how they are written in the CSV
reformat_tips <- function(tree) {
  #Replace "_" with spaces in tree tip labels
  tree$tip.label <- str_replace_all(tree$tip.label, '_', ' ')
  
  #Replace curly brackets with parentheses
  tree$tip.label <- str_replace_all(tree$tip.label, '\\{', '(')
  tree$tip.label <- str_replace_all(tree$tip.label, '\\}', ')')
  
  return(tree)
}


# Error handling
tryCatch({
    # Read the arguments from the command line
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) != 2) {
        stop("Please provide exactly two Newick tree strings.")
    }

    # Parse the Newick tree strings using 'ape'
    tree <- read.tree(file=args[1])

    # Reformat tips
    tree <- reformat_tips(tree)

    # Plot tree and save to png
    png_plot_path = args[2]
    png(filename=png_plot_path, width=1000, height=1350)
    plotTree(ladderize(tree))
    dev.off()


}, error = function(e) {
    cat("Error: ", e$message, "\n")
    quit(status = 1)  # Return an error status code
})