# Подключение модулей
import numpy as np                                    # Для работы с тензорами
from tensorflow.keras.layers import Dense             # Один из типов слоёв в Keras для создания слоя нейронов
from tensorflow.keras.models import Sequential        # Для создания простых моделей используют Sequential,
                                                      # При использовании Sequential достаточно добавить несколько своих слоев

def relu(x):
    return np.maximum(x, 0.)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logical_expression(a, b, c):
    return (a or b) and (b or c)                      # Заданное логическое выражение

dataset = np.array([[0, 0, 0],                        # Все возможные комбинации
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])

def correct_result(dataset):
    return np.array([int(logical_expression(arr[0], arr[1], arr[2])) for arr in dataset])

# Функция, в которой все операции реализованы как поэлементные операции над тензорами
def function_1(dataset, weights):
    dataset = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)                            # Добавляет элемент в конец списка
    for w in range(len(weights)):
        result = np.zeros((len(dataset), len(weights[w][1])))
        for i in range(len(dataset)):
            for j in range(len(weights[w][1])):
                sum = 0
                for k in range(len(dataset[i])):
                    sum += dataset[i][k] * weights[w][0][k][j]
                result[i][j] = layers[w](sum + weights[w][1][j])
        dataset = result
    return dataset

# Функция, в которой все операции реализованы с использованием операций над тензорами из NumPy
def function_2(dataset, weights):
    dataset = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for i in range(len(weights)):
        dataset = layers[i](np.dot(dataset, weights[i][0]) + weights[i][1])
    return dataset

def conclusion(model, dataset):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    correct_res = correct_result(dataset)
    element_wise_res = function_1(dataset, weights)
    numpy_res = function_2(dataset, weights)
    model_res = model.predict(dataset)
    print("_______________________________")
    print("Правильный результат:\n", correct_res)
    print("Результат поэлементых операций:\n", element_wise_res.round(0))
    print("Результат работы с Numpy:\n", numpy_res.round(0))
    print("Результат работы модели:\n", model_res.round(0))
    print("_______________________________")

# Создание модели
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Инициализация параметров обучения
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
conclusion(model, dataset)

# Обучение сети
model.fit(dataset, correct_result(dataset), epochs=150, batch_size=1)
conclusion(model, dataset)