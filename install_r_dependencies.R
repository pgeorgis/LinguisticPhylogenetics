#!/usr/bin/env Rscript
source("phyloLing/utils/r/dependencies.R")

all_dependencies <- c(
	"phytools",
	"ape",
	"stringr",
	"ggplot2",
	"TreeDist"
)

all_bioc_manager_dependencies <- c(
	"ggtree"
)

install_packages_if_needed(all_dependencies)
install_bioc_manager_packages_if_needed(all_bioc_manager_dependencies)
