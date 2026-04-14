import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

with open("model_info.txt", "w") as f:
    f.write(f"local_run_{int(np.random.timestamp()) if hasattr(np, 'timestamp') else 12345}")

print(f"Training complete. Accuracy: {accuracy:.4f}")
print("model_info.txt created")
