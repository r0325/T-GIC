# R implementation of Application 1 of the AIC and BIC methods on FTSE 100 data using AR model with Gaussian noise.
# R implementation of Application 1 for generating Figure 1.
library(ggplot2)

# Calculate logged returns
calculate_log_returns <- function(file_path) {
  # load data
  df <- read.csv(file_path, stringsAsFactors = FALSE)
  df$Close <- as.numeric(df$Close)
  df <- na.omit(df)  
  df$Close <- rev(df$Close)
  # logged returns
  df$log_returns <- c(NA, diff(log(df$Close)))
  df <- na.omit(df)
  return(df)
}

# FTSE 100 data
file_path <- "D:/R/FTSE100data.csv"  
df <- calculate_log_returns(file_path)

# Plot (Figure 1)
index_ticks <- seq(1, nrow(df), length.out=4)
index_ticks
custom_labels <- c("1986/01/02", "1997/11/06", "2009/09/21", "2021/04/08")
plot(df$log_returns, type="l", xlab="Time", ylab="Logged Returns", xaxt="n",
     main = "FTSE 100")
axis(1, at=index_ticks, labels=custom_labels)

# AIC and BIC values of AR models
vw <- df$log_returns
aic_values <- numeric(10)
bic_values <- numeric(10)
k_values <- 1:10

for (k in k_values) {
  model <- arima(vw, order = c(k, 0, 0))
  aic_values[k] <- model$aic
  bic_values[k] <- AIC(model, k = log(length(vw)))
}
print("aic_values")
print(aic_values)
print("bic_values")
print(bic_values)

# Model selection by AIC and BIC
best_k_aic <- which.min(aic_values)
best_k_bic <- which.min(bic_values)
cat("minimum AIC: ", min(aic_values), "k: ", best_k_aic, "\n")
cat("minimum BIC: ", min(bic_values), "k: ", best_k_bic, "\n")

cat("ARIMA(", best_k_aic, ", 0, 0) parameter estimates: \n")
print(arima(vw, order = c(best_k_aic, 0, 0))$coef)
print(arima(vw, order = c(best_k_aic, 0, 0))$sigma2)
cat("ARIMA(", best_k_bic, ", 0, 0) parameter estimates: \n")
print(arima(vw, order = c(best_k_bic, 0, 0))$coef)
print(arima(vw, order = c(best_k_bic, 0, 0))$sigma2)
