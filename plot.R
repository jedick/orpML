# orpML/plot.R
# Plot output from ML pipeline in Python
# 20241125 jmd

# Plot score vs number of components in PCA for models with one and two feature sets
plot_n_components <- function() {
  dat1 <- read.csv("results/n_components/histgbr_1.csv")
  dat2 <- read.csv("results/n_components/histgbr_2.csv")
  ylim <- -rev(range(c(dat1$mean_test_score, dat2$mean_test_score), na.rm=TRUE))
  plot(dat1$param_reduce_dim__n_components, -dat1$mean_test_score, type = "l", lty = 2, ylim = ylim)
  lines(dat2$param_reduce_dim__n_components, -dat2$mean_test_score, type = "l")
}

# Plot predicted vs actual Eh7 for test data
plot_test <- function() {
  dat <- read.csv("results/test_results.csv")
  xylim <- range(dat$Eh7, dat$Eh7_pred)
  plot(dat$Eh7, dat$Eh7_pred, xlab = "Measured Eh7 (mV)", ylab = "Predicted Eh7 (mV)", pch = 19, col = "#00000080", xlim = xylim, ylim = xylim)
  # Calculate MAE
  MAE <- mean(abs(dat$Eh7 - dat$Eh7_pred))
  # Fit linear model to calculate R2
  thislm <- lm(Eh7_pred ~ Eh7, dat)
  R2 <- summary(thislm)$r.squared
  # Plot best-fit line
  x <- c(-1000, 1000)
  y <- predict(thislm, data.frame(Eh7 = x))
  lines(x, y, col = 2, lty = 2, lwd = 2)
  # Plot 1:1 line
  lines(x, x, col = 8, lty = 2, lwd = 2)
}
