# R implementation of Application 3 for generating Figure 3.
library(ggplot2)

# Data process
process_wind_direction <- function(file1, file2) {
  df1 <- read.csv(file1)
  df2 <- read.csv(file2)
  direction_to_radian <- list(
    "N" = 0,
    "NNE" = pi / 8,
    "NE" = 2 * pi / 8,
    "ENE" = 3 * pi / 8,
    "E" = 4 * pi / 8,
    "ESE" = 5 * pi / 8,
    "SE" = 6 * pi / 8,
    "SSE" = 7 * pi / 8,
    "S" = 8 * pi / 8,
    "SSW" = 9 * pi / 8,
    "SW" = 10 * pi / 8,
    "WSW" = 11 * pi / 8,
    "W" = 12 * pi / 8,
    "WNW" = 13 * pi / 8,
    "NW" = 14 * pi / 8,
    "NNW" = 15 * pi / 8
  )
  x1 <- sapply(df1$value, function(x) direction_to_radian[[x]])
  x2 <- sapply(df2$value, function(x) direction_to_radian[[x]])
  samples <- data.frame(x1 = x1, x2 = x2)
  return(samples)
}

# 2-D histogram (Figure 3)
plot_2d_histogram <- function(samples) {
  ggplot(samples, aes(x = x1, y = x2)) +
    geom_bin2d(bins = 30, color = "white") +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(x = "Wind Direction (0:00)", y = "Wind Direction (12:00)") +
  theme_minimal()
}

samples <- process_wind_direction('D:/R/wind_data_value_2023_00.csv', 'D:/R/wind_data_value_2023_12.csv')
plot_2d_histogram(samples)

