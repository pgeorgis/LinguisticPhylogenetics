#!/usr/bin/env Rscript

# Function to silently check if a package is installed
is_package_installed <- function(package_name) {
	return(require(package_name, quietly = TRUE, character.only = TRUE))
}

# Function to load a package without printing its startup messages
load_package <- function(package_name) {
	suppressPackageStartupMessages(library(package_name, quietly = TRUE, character.only = TRUE))
}

# Function to check if a package is installed and install it if necessary
install_if_needed <- function(package_name) {
    if (!is_package_installed(package_name)) {
        install.packages(package_name, repos = "https://cran.r-project.org")
    }
	load_package(package_name)
}

# Function to check if a Bioconductor package is installed and install it if necessary
install_if_needed_using_bioc_manager <- function(package_name) {
	install_if_needed("BiocManager")

	if (!is_package_installed(package_name)) {
		BiocManager::install(package_name, quiet = TRUE)
	}
	load_package(package_name)
}

# Function to install a list of packages if they are not already installed
install_packages_if_needed <- function(packages) {
	for (package in packages) {
		install_if_needed(package)
	}
}

# Function to install a list of Bioconductor packages if they are not already installed
install_bioc_manager_packages_if_needed <- function(packages) {
	for (package in packages) {
		install_if_needed_using_bioc_manager(package)
	}
}
