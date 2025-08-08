import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)

print("Predictions:", clf.predict(X))

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', label='0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', label='1' if i == 1 else "")

if clf.coef_[0][1] != 0:
    x_values = np.array([0, 1])
    y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_values, y_values, label='Decision Boundary')
else:
    print("Cannot plot decision boundary (vertical line or undefined slope)")

plt.title('Perceptron Decision Boundary for XOR')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()


Output:


<img width="843" height="581" alt="Screenshot 2025-08-08 210336" src="https://github.com/user-attachments/assets/9a334759-ab2e-4d81-a7dc-38079fc99fca" />
