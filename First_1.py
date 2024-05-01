import numpy as np  # Імпортуємо бібліотека для багатовимірних масивів.


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Оголошуємо нашу функцію (сигмоїд) та беремо з неї "exp" (2.7)


training_inputs = np.array([[0, 0, 1],  # Наші вхідні дані у вигляді масиву
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T  # Очікувані результати + "Т" - транспонований масив

np.random.seed(1)  # Ініціалізуємо ваги за допомогою рандомного генератора подій

synaptic_weights = 2 * np.random.random((3, 1)) - 1  # Вага синапсу не може бути меншою за нуль і більше одиниці

print("Випадкові ваги, що ініціалізують")
print(synaptic_weights)

# Спробуємо навчити нашу нейронну мережу за допомогою методу зворотного розповсюдження
for i in range(30000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustments

print("Ваги після навчання:")
print(synaptic_weights)

print("Результат після навчання:")
print(outputs)

###############################################################################
new_inputs = np.array([0, 0, 0])  # New Situation
output = sigmoid(np.dot(new_inputs, synaptic_weights))  # Викликаємо нашу функцію

print("Нова ситуація:")
print(output)