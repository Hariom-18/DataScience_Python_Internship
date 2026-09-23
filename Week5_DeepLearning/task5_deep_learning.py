import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

# 1. Dataset Loading
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Data Preprocessing & Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Model Architecture Design (Sequential Artificial Neural Network)
model = Sequential(
    [
        Flatten(input_shape=(28, 28), name="Input_Flatten_Layer"),
        Dense(128, activation="relu", name="Hidden_Layer_1"),
        Dropout(0.2, name="Dropout_Layer"),  # Address Overfitting
        Dense(64, activation="relu", name="Hidden_Layer_2"),
        Dense(10, activation="softmax", name="Output_Layer"),
    ]
)

# Model Summary
model.summary()

# 4. Compilation with Hyperparameters
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 5. Model Training
history = model.fit(
    X_train, y_train, epochs=10, validation_split=0.2, batch_size=64
)

# 6. Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Visualizations (Training Curves & Confusion Matrix)
plt.figure(figsize=(14, 5))

# Plot 1: Loss & Accuracy History
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", lw=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", lw=2)
plt.title("Model Accuracy across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Plot 2: Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (MNIST Digits 0-9)")
plt.xlabel("Predicted Digit")
plt.ylabel("True Digit")

plt.tight_layout()
plt.savefig("deep_learning_visualizations.png")
plt.show()

print("Model execution completed! Saved chart as 'deep_learning_visualizations.png'")