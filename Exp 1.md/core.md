from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, epochs=1000, verbose=0)

loss, acc = model.evaluate(X, Y, verbose=0)
print("Accuracy:", acc)

predictions = model.predict(X)
print("Predictions:\n", predictions)
print("Rounded Predictions:\n", np.round(predictions))

plt.plot(history.history['loss'])
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)<img width="675" height="779" alt="Screenshot 2025-08-08 211100" src="https://github.com/user-attachments/assets/7ad2f6e3-efb5-4765-8d6f-1578ec280c02" />

plt.show()
