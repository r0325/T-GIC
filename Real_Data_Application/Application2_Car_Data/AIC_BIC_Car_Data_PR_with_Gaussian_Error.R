# R implementation of Application 2 of the AIC and BIC methods on Car data using polynomial regression model with Gaussian error.
library(ISLR)

data(Auto)
Auto$horsepower <- scale(Auto$horsepower)
Auto$mpg <- log(Auto$mpg)

# Model selection by AIC and BIC
aic_bic_selection <- function(data, max_degree = 5) {
  results <- data.frame(Degree = integer(0), AIC = numeric(0), BIC = numeric(0))
  
  for (i in 1:max_degree) {
    model <- lm(mpg ~ poly(horsepower, i,raw = TRUE), data = data)
    aic_val <- AIC(model)
    bic_val <- BIC(model)
    results <- rbind(results, data.frame(Degree = i, AIC = aic_val, BIC = bic_val))
  }
  
  return(results)
}

results <- aic_bic_selection(Auto, max_degree = 10)
print(results)

# Best degree order selected by AIC and BIC
best_aic_degree <- results$Degree[which.min(results$AIC)]
best_bic_degree <- results$Degree[which.min(results$BIC)]
cat("Best degree based on AIC: ", best_aic_degree, "\n")
cat("Best degree based on BIC: ", best_bic_degree, "\n")

# Model fitted by AIC 
best_model_aic <- lm(mpg ~ poly(horsepower, best_aic_degree,raw = TRUE), data = Auto)
cat("Best model based on AIC:\n")
summary(best_model_aic)
residual_variance_aic <- var(residuals(best_model_aic))
residual_variance_aic

# Model fitted by BIC
best_model_bic <- lm(mpg ~ poly(horsepower, best_bic_degree,raw = TRUE), data = Auto)
cat("Best model based on BIC:\n")
summary(best_model_bic)
residual_variance_bic <- var(residuals(best_model_bic))
residual_variance_bic
