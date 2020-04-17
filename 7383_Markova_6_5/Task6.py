# Подключение модулей
import numpy as np                                    # Предоставляет общие математические и числовые операции
from sklearn.preprocessing import LabelEncoder        # Для преобразования категорий в понятные модели числовые данные
from tensorflow.keras.models import Sequential        # Для создания простых моделей используют Sequential,
                                                      # При использовании Sequential достаточно добавить несколько своих слоев
from sklearn.model_selection import train_test_split  # Деление выборки на тренировочную и тестовую часть
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

from var5 import gen_data

# Загрузка данных
size = 13000
dataset, labels = gen_data(size)                      # dataset - входные данные, labels - выходные данные
dataset = np.asarray(dataset)
labels = np.asarray(labels)

dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.2)
dataset_train = dataset_train.reshape(dataset_train.shape[0], 50, 50, 1)
dataset_test = dataset_test.reshape(dataset_test.shape[0], 50, 50, 1)

dataset_train = dataset_train.astype('float32')
dataset_test = dataset_test.astype('float32')
# Нормализация
dataset_train /= 255
dataset_test /= 255

# Переход от текстовых меток к категориальному вектору (Вектор -> матрица)
encoder = LabelEncoder()
encoder.fit(labels_train)                             # Метод fit этого класса находит все уникальные значения и строит таблицу для соответствия каждой категории некоторому числу
labels_train = encoder.transform(labels_train)        # Метод transform непосредственно преобразует значения в числа

encoder.fit(labels_test)
labels_test = encoder.transform(labels_test)

n = len(dataset_train)//5

dataset_validation = dataset_train[:n]
labels_validation = labels_train[:n]
dataset_train = dataset_train[n:]
labels_train = labels_train[n:]

# Создание модели
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(Dropout(0.25))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Инициализация параметров обучения
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение сети
hist = model.fit(dataset_train, labels_train, epochs=10, batch_size=32, validation_data=(dataset_validation, labels_validation))

model.evaluate(dataset_test, labels_test)