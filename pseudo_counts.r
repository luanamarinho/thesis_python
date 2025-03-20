library(ggplot2)
sce <- readRDS("C:\\Users\\luana\\Documents\\thesis\\data\\sce.rds")
dec <- scran::modelGeneVarWithSpikes(sce, "ERCC")
genes_to_use <- scran::getTopHVGs(dec, prop = 0.1)

effectLogTransform <- lapply(c(TRUE, FALSE), function(log_transform) {
  sceData <- scuttle::logNormCounts(sce, pseudo.count = 0.01, log = log_transform)
  assay_name <- ifelse(log_transform, "logcounts", "normcounts")
  out <- assay(sceData, assay_name)
  rm(sceData)
  return(out)
}) |> setNames(c("logcounts", "normcounts"))
median_SD_hvg <- lapply(effectLogTransform, function(norm_x) {
  median_genes <- apply(norm_x[genes_to_use, ], MARGIN = 1, median, na.rm = TRUE)
  sd_genes <- apply(norm_x[genes_to_use, ], MARGIN = 1, sd, na.rm = TRUE)
  return(data.frame("median" = median_genes, "sd" = sd_genes))
})

hist(median_SD_hvg$normcounts$median, breaks = 50, main = "Median of HVGs", xlab = "Median")
hist(median_SD_hvg$logcounts$median, breaks = 50, main = "Median of HVGs", xlab = "Median")

effectPseudoCount <- lapply(c(0.01, 0.1, 1), function(pseudo_count) {
  sceData <- scuttle::logNormCounts(sce, pseudo.count = pseudo_count, log = TRUE)
  out <- assay(sceData, "logcounts")
  rm(sceData)
  return(out)
}) |> setNames(paste0("pseudo_", c(0.01, 0.1, 1)))

median_SD_pseudo <- lapply(effectPseudoCount, function(norm_x) {
  median_genes <- apply(norm_x[genes_to_use, ], MARGIN = 1, median, na.rm = TRUE)
  sd_genes <- apply(norm_x[genes_to_use, ], MARGIN = 1, sd, na.rm = TRUE)
  return(data.frame("median" = median_genes, "sd" = sd_genes))
})

histogram_plot_median <- lapply(names(median_SD_pseudo), function(pseudo) {
  ggplot(median_SD_pseudo[[pseudo]], aes(x = median)) +
    geom_histogram(bins = 50, fill = "blue", alpha = 0.7) +
    ggtitle(paste("Distribution of the median of HVGs", pseudo, ")")) +
    xlab("Median") +
    theme_minimal()
})
gridExtra::grid.arrange(grobs = histogram_plot_median, ncol = 3)

histogram_plot_sd <- lapply(names(median_SD_pseudo), function(pseudo) {
  ggplot(median_SD_pseudo[[pseudo]], aes(x = sd)) +
    geom_histogram(bins = 50, fill ="blue", alpha = 0.7) +
    xlab(pseudo) +
    ylab("") +
    theme_minimal()
})
grid_sd <- gridExtra::grid.arrange(grobs = histogram_plot_sd, ncol = 3)

ggsave("C:\\Users\\luana\\Documents\\thesis\\plots\\sd_histograms.png", grid_sd, width = 15, height = 5)
