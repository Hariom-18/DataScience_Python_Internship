import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for plots
sns.set_theme(style="whitegrid")

# 1. Load Dataset (Titanic Dataset)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Summary Statistics
print("--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics (Numerical) ---")
print(df.describe())

# 3. Visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Survival Count
plt.subplot(2, 2, 1)
sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Overall Survival Count (0 = No, 1 = Yes)")
plt.xlabel("Survived")
plt.ylabel("Count")

# Plot 2: Survival by Gender
plt.subplot(2, 2, 2)
sns.countplot(x="Survived", hue="Sex", data=df, palette="Set1")
plt.title("Survival Count by Gender")
plt.xlabel("Survived")
plt.ylabel("Count")

# Plot 3: Age Distribution
plt.subplot(2, 2, 3)
sns.histplot(df["Age"].dropna(), kde=True, color="teal", bins=30)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")

# Plot 4: Correlation Heatmap
plt.subplot(2, 2, 4)
numeric_df = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_visualizations.png")
plt.show()

print("\nVisualizations saved successfully as 'eda_visualizations.png'!")