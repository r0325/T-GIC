# R implementation of Application 2 for generating Figure 2.
library(ISLR)

data(Auto)
Auto$horsepower <- scale(Auto$horsepower)
Auto$mpg <- log(Auto$mpg)

# Plots of logged mpg versus standardized horsepower (Figure 2)
plot(mpg~horsepower, data=Auto,
     main = "Plots and the Fits Comparison",
     xlab = "Standardized Horsepower", ylab = "Logged MPG")

# Fitted polynomial regression models
plot_poly2 <- function(coef, color,lty_num) {
  curve(coef[1] + coef[2] * x + coef[3] * x^2, 
        add = TRUE, col = color, lwd = 2,lty=lty_num)
}

plot_poly7 <- function(coef, color,lty_num) {
  curve(coef[1] + coef[2] * x + coef[3] * x^2 + 
          coef[4] * x^3+ coef[5] * x^4 +
          coef[6] * x^5+ coef[7] * x^6 + coef[8] * x^7, 
        add = TRUE, col = color, lwd = 2,lty=lty_num)
}

# Parameter estimates
coef_GIC <- c(3.0287626955229086,-0.38378966,0.07995221) 
coef_BIC2 <- c(3.040667,-0.344804,0.057793)
coef_AIC7 <- c(3.037590,-0.296597,0.072299,-0.123224,0.041173,0.048936,-0.032108,0.005132)

# Plots of fitted polynomial regression models (Figure 2)
plot_poly7(coef_AIC7,"green",1)
plot_poly2(coef_BIC2, "blue",1)
plot_poly2(coef_GIC, "red",2)
legend("topright", 
       legend = c("7th-degree model (Gaussian errors)",
                  "2nd-degree model (Gaussian errors)",  
                  "2nd-degree model (Nxt errors)"), 
       col = c("green","blue","red"), lty = c(1,1,2), lwd = 2,
       bty = "o",inset = c(0.01, 0.01))
