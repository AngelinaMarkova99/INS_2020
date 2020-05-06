# Подключение модулей
import numpy as np                                    # Предоставляет общие математические и числовые операции
import pandas as pd                                   # Инструмент для анализа и обработки данных
import matplotlib.pyplot as plt                       # Библиотека для визуализации графиков
import tensorflow.keras as keras
from sklearn import preprocessing
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

from var5 import gen_data

class Classification(keras.Model):
    def __init__(self, num_classes=2):
        super(Classification, self).__init__()

        self.initializer = initializers.normal
        self.num_classes = num_classes

        # Создание модели
        self.features = Sequential([
            Conv2D(filters=128, kernel_size=5, strides=1, activation='relu', use_bias=True),
            Dropout(0.25),
            Conv2D(filters=64, kernel_size=5, strides=1, activation='relu', use_bias=True),
            Dropout(0.3),
            Conv2D(filters=128, kernel_size=5, strides=1, activation='relu'),
            Dense(50, activation='relu'),
        ])

        self.lin = Sequential([
            Flatten(),
            Dense(units=self.num_classes, activation='softmax'),
        ])

    def call(self, inputs):
        x = self.features(inputs)
        print(x.shape)
        x = self.lin(x)
        return x

epochs = 15
batch_size = 10

# Реализация собственного CallBack (обратного вызова)
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, y_train, interval=3, epochs=epochs):
        super(CustomCallback, self).__init__()
        self.y_train = y_train
        self.epochs = epochs
        self.push_interval = interval

    def on_train_begin(self, logs=None):
        self.sample = []

    def on_epoch_begin(self, epoch, logs=None):
        self.worst_batch_index = 0
        self.worst_batch_acc = 1.0
        self.worst_batch_loss = 0
        self.worst_batch_class = 0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.push_interval == 0 or epoch == 0 or (epoch+1) == self.epochs:
            epoch_data = [(epoch+1)]
            epoch_data.append(self.worst_batch_index)
            epoch_data.append(self.worst_batch_acc)
            epoch_data.append(self.worst_batch_loss)
            epoch_data.append(self.worst_batch_class)
            self.sample.append(epoch_data)

    def on_batch_end(self, batch, logs=None):
        acc = logs.get('acc')
        if self.worst_batch_acc > float(acc):
            self.worst_batch_acc = acc
            self.worst_batch_index = batch
            self.worst_batch_loss = logs.get('loss')
            self.worst_batch_class = self.y_train[batch]

# Загрузка данных
X, Y = gen_data()

X = np.asarray(X)
Y = np.asarray(Y).flatten()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

print(y_test[0:10])

y_train_non_categ = y_train

# Переход от текстовых меток к категориальному вектору (Вектор -> матрица)
encoder = preprocessing.LabelEncoder()
encoder.fit(Y)

y_test = np.asarray(encoder.transform(y_test))
y_train = np.asarray(encoder.transform(y_train))

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# Нормализация
x_train = np.asarray(x_train) / 255
x_test = np.asarray(x_test) / 255

x_train = np.asarray(x_train).reshape(400, 50, 50, 1)
x_test = np.asarray(x_test).reshape(100, 50, 50, 1)

classifier = Classification()
criterion = losses.CategoricalCrossentropy()

history = CustomCallback(y_train_non_categ)

# Инициализация параметров обучения
classifier.compile(optimizer=optimizers.Adam(lr=0.0001), loss=criterion, metrics=['accuracy'])
# Обучение сети
H = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[history])

test_loss, test_acc = classifier.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# История обратных вызовов
table = np.asarray(history.sample)
df = pd.DataFrame(table, columns=['epoch', 'batch_num', 'accuracy', 'loss', 'truth_class'])
df.to_csv('Result_table.csv', index=False)

# Построение графиков ошибки и точности
plt.figure(1, figsize=(8, 5))
plt.title("Model accuracy")
plt.plot(H.history['acc'], 'c', label='Train')
plt.plot(H.history['val_acc'], 'g', label='Test')
plt.legend()
plt.grid()
plt.show()
plt.clf()

plt.figure(1, figsize=(8, 5))
plt.title("Model loss")
plt.plot(H.history['loss'], 'c', label='Train')
plt.plot(H.history['val_loss'], 'g', label='Test')
plt.legend()
plt.grid()
plt.show()
plt.clf() 