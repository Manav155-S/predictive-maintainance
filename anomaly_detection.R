# Install required packages
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if (!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

# Load libraries
library(ggplot2)
library(dplyr)

# Read the CSV file from local path
file_path <- "predictive_maintenance.csv"
df <- read.csv(file_path, check.names = FALSE)

# View column names to confirm
print(colnames(df))

# Convert required columns to numeric (safely)
df$`Torque [Nm]` <- as.numeric(as.character(df$`Torque [Nm]`))
df$`Rotational speed [rpm]` <- as.numeric(as.character(df$`Rotational speed [rpm]`))

# Drop NA rows (due to non-numeric conversion or missing values)
df <- na.omit(df)

# Check if column has numeric data now
if (all(is.na(df$`Torque [Nm]`))) {
  stop("Torque column still contains only NA. Check if the column values are properly formatted.")
}

# Z-score calculation
df$zscore <- (df$`Torque [Nm]` - mean(df$`Torque [Nm]`)) / sd(df$`Torque [Nm]`)

# Filter anomalies
anomalies <- df %>% filter(abs(zscore) > 2)

# Save anomaly points
write.csv(anomalies, "anomalies_output.csv", row.names = FALSE)

# Save the plot
png("anomaly_plot.png", width = 800, height = 600)
ggplot(df, aes(x = `Rotational speed [rpm]`, y = `Torque [Nm]`)) +
  geom_point(color = "blue") +
  geom_point(data = anomalies, aes(x = `Rotational speed [rpm]`, y = `Torque [Nm]`), color = "red") +
  theme_minimal() +
  labs(title = "Anomaly Detection using Z-Score",
       x = "Rotational Speed (rpm)",
       y = "Torque (Nm)")

# Save anomaly data
write.csv(anomalies, "anomalies_output.csv", row.names = FALSE)

# Create a summary
summary_text <- paste("✅ Anomaly Detection Complete.\nTotal records:", nrow(df),
                      "\nAnomalies found:", nrow(anomalies), "\nTimestamp:", Sys.time())

writeLines(summary_text, "summary.txt")


