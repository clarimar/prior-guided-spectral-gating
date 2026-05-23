# Download Tecator dataset
install.packages('pls', repos='http://cran.us.r-project.org')
library(pls)
data(meatspec)

output_dir <- '~/Dropbox/chemometric-band-selection/data/raw/tecator'
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

write.csv(meatspec$X, file.path(output_dir, 'spectra.csv'), row.names=FALSE)
write.csv(meatspec$Y, file.path(output_dir, 'targets.csv'), row.names=FALSE)

cat('✅ Arquivos salvos em:', output_dir, '\n')
