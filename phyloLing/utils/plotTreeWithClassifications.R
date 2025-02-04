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

install_if_needed("phytools")
install_if_needed("ape")
install_if_needed("stringr")
install_if_needed("BiocManager")
install_if_needed("ggtree")
install_if_needed("ggplot2")

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


max_distance_newick <- function(newick_str) {
  # Compute the root-to-tip distances
  distances <- node.depth.edgelength(tree)
  
  # Return the maximum distance
  return(max(distances))
}


# Error handling
tryCatch({
    # Read the arguments from the command line
    args <- commandArgs(trailingOnly = TRUE)

    if (length(args) != 3) {
        stop("Please provide 1) path to file containing Newick string, 2) path to PNG outfile, 3) path to classification CSV.")
    }

    # Parse the Newick tree strings using 'ape'
    tree <- read.tree(file=args[1])

    # Reformat tips
    tree <- reformat_tips(tree)

    # Get maximum depth
    max_depth <- max_distance_newick(tree)

    # Load classification file
    classification_data <- read.csv(args[3], sep=',')

    # Plot tree and save to png
    png_plot_path = args[2]
    png(filename=png_plot_path, width=1000, height=1350)
    p <- ggtree(tree, ladderize = FALSE) %<+% classification_data + 
        geom_tippoint(aes(color=Classification), size=5, show.legend=FALSE) + 
        geom_tiplab(aes(color=Classification), offset=0.01, align=FALSE, show.legend=FALSE, size=9, fontface='bold.italic', family='Times') +
        #theme(legend.text=element_text(size=11, family="Times"), legend.title=element_text(size=14, family="Times", face='bold')) +
        xlim(0, max_depth + (max_depth * 0.5))
    print(p)
    dev.off()

}, error = function(e) {
    cat("Error: ", e$message, "\n")
    quit(status = 1)  # Return an error status code
})