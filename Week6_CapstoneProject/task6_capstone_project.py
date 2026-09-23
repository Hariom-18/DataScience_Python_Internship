import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Data Acquisition
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Data Cleaning & Preprocessing (Fixed Pandas 3.0+ Syntax)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# 3. Feature Selection & Scaling
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_Q",
    "Embarked_S",
]
X = df[features]
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Unsupervised Learning (K-Means Profiling)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster_Group"] = kmeans.fit_predict(X_scaled)

# 5. Supervised Learning (Random Forest Classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Performance Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"End-to-End Model Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Comprehensive Multi-Panel Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Feature Correlation Heatmap
sns.heatmap(
    df[features + ["Survived"]].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    ax=axes[0, 0],
)
axes[0, 0].set_title("1. Feature Correlation Heatmap")

# Subplot 2: K-Means Segmentation
sns.scatterplot(
    x=X_scaled[:, 2],
    y=X_scaled[:, 5],
    hue=df["Cluster_Group"],
    palette="Set2",
    ax=axes[0, 1],
)
axes[0, 1].set_title("2. Unsupervised K-Means Customer Groups (Age vs Fare)")

# Subplot 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes[1, 0])
axes[1, 0].set_title("3. Random Forest Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# Subplot 4: Feature Importance Ranking
importances = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=True
)
importances.plot(kind="barh", color="skyblue", ax=axes[1, 1])
axes[1, 1].set_title("4. Supervised Model Feature Importances")

plt.tight_layout()
plt.savefig("capstone_comprehensive_analysis.png")
plt.show()

print(
    "Capstone Pipeline Executed Successfully! Chart saved as 'capstone_comprehensive_analysis.png'"
)