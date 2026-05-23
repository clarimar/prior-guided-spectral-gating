#!/usr/bin/env Rscript

cat("=======================================================\n")
cat("LISTANDO DATASETS DISPONÍVEIS\n")
cat("=======================================================\n\n")

# pls package
cat("1. PLS PACKAGE:\n")
library(pls)
data_pls <- data(package="pls")
print(data_pls$results[, c("Item", "Title")])

# Verificar tamanho dos principais
cat("\n\n2. DATASET DETAILS:\n\n")

# Gasoline
data(gasoline)
cat("GASOLINE:\n")
cat("  Spectra:", dim(gasoline$NIR), "\n")
cat("  Target: octane rating\n\n")

# Yarn
data(yarn)
cat("YARN:\n")
cat("  Spectra:", dim(yarn$NIR), "\n")
cat("  Target: density\n\n")

# Oliveoil
data(oliveoil)
cat("OLIVEOIL:\n")
cat("  Spectra:", dim(oliveoil$sensory), "\n\n")

cat("=======================================================\n")
