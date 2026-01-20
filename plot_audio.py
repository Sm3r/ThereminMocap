import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("data/dataframes/TestZED.csv")

# Create the plot
plt.figure(figsize=(12, 6))

# Plot Pitch_CV and Volume_CV with different colors
plt.plot(df['Frame'], df['Pitch_CV'], color='blue', label='Pitch CV', linewidth=0.5)
plt.plot(df['Frame'], df['Volume_CV'], color='red', label='Volume CV', linewidth=0.5)

# Add labels and title
plt.xlabel('Frame')
plt.ylabel('CV Value')
plt.title('Pitch CV and Volume CV over Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
