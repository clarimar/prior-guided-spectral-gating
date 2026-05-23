#!/usr/bin/env Rscript

library(pls)
data(gasoline)

cat("Verificando estrutura do dataset...\n")
cat(sprintf("NIR dim: %d x %d\n", nrow(gasoline$NIR), ncol(gasoline$NIR)))

# Criar diretório
dir.create("data/raw/gasoline", recursive = TRUE, showWarnings = FALSE)

# Extrair matriz
spectra_matrix <- as.matrix(gasoline$NIR)

# Salvar diretamente com write.table
colnames(spectra_matrix) <- paste0("Band_", 1:ncol(spectra_matrix))
write.table(spectra_matrix, 
            "data/raw/gasoline/spectra.csv", 
            sep = ",", 
            row.names = FALSE,
            col.names = TRUE)

# Salvar targets
write.table(data.frame(octane = as.numeric(gasoline$octane)),
            "data/raw/gasoline/targets.csv",
            sep = ",",
            row.names = FALSE,
            col.names = TRUE)

cat("\n✅ Gasoline dataset saved!\n")
cat(sprintf("   Samples: %d\n", nrow(spectra_matrix)))
cat(sprintf("   Bands: %d\n", ncol(spectra_matrix)))
cat(sprintf("   Target range: %.2f - %.2f\n", 
            min(gasoline$octane), max(gasoline$octane)))

# Verificar
test <- read.csv("data/raw/gasoline/spectra.csv")
cat(sprintf("\nVerified: %d x %d\n", nrow(test), ncol(test)))
