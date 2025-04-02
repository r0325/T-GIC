# R implementation of Application 1 for the rolling m-step-ahead forecast on FTSE 100 data using AR model with Gaussian noise.
library(forecast)

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
vw <- df$log_returns

train_size <- length(vw) - 100  
train_data <- vw[1:train_size]
test_data <- vw[(train_size + 1):length(vw)]

# Rolling m-step-ahead forecast
m_values <- 1:5
mse_results <- data.frame(m = m_values, MSE = numeric(length(m_values)))  

for(m in m_values) {
  forecasts <- numeric(length = length(test_data) - m + 1)  
  actual_values <- test_data[m:length(test_data)]  
  for(i in 1:(length(test_data) - m + 1)) {
    model <- arima(vw[i:(train_size + i - 1)], order = c(6, 0, 0))  # AR(6) model
    # model <- arima(vw[i:(train_size + i - 1)], order = c(7, 0, 0)) # AR(7) model
    forecasts[i] <- predict(model, n.ahead = m)$pred[m]
  }
  mse_results$MSE[mse_results$m == m] <- mean((actual_values - forecasts)^2)
}

print(mse_results)
