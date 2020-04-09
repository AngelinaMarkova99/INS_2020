# Подключение модулей
import numpy as np                                    # Общие математические и числовые операции
from PIL import Image                                 # Библиотека для обработки графики
from numpy import asarray
import matplotlib.pyplot as plt                       # Библиотека для визуализации графиков
from keras.datasets import mnist                      # База данных рукописных цифр
from keras.utils import to_categorical                # Преобразует вектор класса (целые числа) в двоичную матрицу классов
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential        # Для создания простых моделей используют Sequential,
                                                      # При использовании Sequential достаточно добавить несколько своих слоев
from tensorflow.keras.layers import Dense, Flatten

# Загрузка набора данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Вывод изображений для тестов в серых оттенках
plt.subplot(221)
plt.imshow(X_test[84], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_test[142], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_test[139], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_test[35], cmap=plt.get_cmap('gray'))
plt.show()

# Сохранение изображений
# temp = np.reshape(X_test[84], (28, 28))
# im = Image.fromarray(temp).convert('L')
# im.save("numeral_8.png")
# temp = np.reshape(X_test[142], (28, 28))
# im = Image.fromarray(temp).convert('L')
# im.save("numeral_3.png")
# temp = np.reshape(X_test[139], (28, 28))
# im = Image.fromarray(temp).convert('L')
# im.save("numeral_4.png")
# temp = np.reshape(X_test[35], (28, 28))
# im = Image.fromarray(temp).convert('L')
# im.save("numeral_2.png")

# Нормализовать входные данные от 0-255 до 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Горячее кодирование значений класса, преобразовывая вектор целых чисел класса в двоичную матрицу
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Определение базовой модели
def baseline_model(optimazers_list, labels):
    acc_list = []
    history_loss_list = []
    history_acc_list = []

    # Создание модели
    for opt in optimazers_list:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # Инициализация параметров обучения
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Обучение сети
        hist = model.fit(X_train, y_train, epochs=5, batch_size=128)
        history_loss_list.append(hist.history['loss'])
        history_acc_list.append(hist.history['acc'])
        test_loss, test_acc = model.evaluate(X_test, y_test)
        acc_list.append(test_acc)

    # Построение графиков ошибки и точности
    print("---------------------------------------------------------")
    print("Точность = " + str(np.round(acc_list, 2)) + " %")
    x = range(1, 6)

    plt.subplot(211)
    plt.title('Loss')
    for loss in history_loss_list:
        plt.plot(x, loss, 'c')
    plt.legend(labels)
    plt.grid()

    plt.subplot(212)
    plt.title('Accuracy')
    for acc in history_acc_list:
        plt.plot(x, acc, 'g')
    plt.legend(labels)
    plt.grid()
    plt.show()

    return model

optimazers_list = []

### Исследование влияния различных оптимизаторов, а также их параметров, на процесс обучения ###

# Оптимизаторы с параметрами по умолчанию
# optimazers_list = ("adam", "adagrad", "rmsprop", "sgd")
# baseline_model(optimazers_list, optimazers_list)

# sgd
# optimazers_list.append(optimizers.SGD())
# optimazers_list.append(optimizers.SGD(learning_rate=0.1, momentum=0.0))
# optimazers_list.append(optimizers.SGD(learning_rate=0.1, momentum=0.8))
# optimazers_list.append(optimizers.SGD(learning_rate=0.01, momentum=0.8))
# baseline_model(optimazers_list, ("SGD(default)", "SGD(learning_rate=0.1, momentum=0.0)", "SGD(learning_rate=0.1, momentum=0.8)", "SGD(learning_rate=0.01, momentum=0.8)"))

# adagrad
# optimazers_list.append(optimizers.Adagrad())                                 # default (0.01)
# optimazers_list.append(optimizers.Adagrad(learning_rate=0.1))
# optimazers_list.append(optimizers.Adagrad(learning_rate=0.001))
# baseline_model(optimazers_list, ("Adagrad(default)", "Adagrad(learning_rate=0.1)", "Adagrad(learning_rate=0.001)"))

# rmsprop
# optimazers_list.append(optimizers.RMSprop())                                 # default learning_rate=0.001, rho=0.9
# optimazers_list.append(optimizers.RMSprop(learning_rate=0.01, rho = 0.9))
# optimazers_list.append(optimizers.RMSprop(learning_rate=0.01, rho = 0.5))
# optimazers_list.append(optimizers.RMSprop(learning_rate=0.001, rho = 0.5))
# baseline_model(optimazers_list, ("RMSprop(default)", "RMSprop(learning_rate=0.01, rho = 0.9)", "RMSprop(learning_rate=0.01, rho = 0.5)", "RMSprop(learning_rate=0.001, rho = 0.5)"))

# adam
# optimazers_list.append(optimizers.Adam())                                    # default (lr=0.001, beta_1=0.9, beta_2=0.999)
# optimazers_list.append(optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=True))
# optimazers_list.append(optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=True))
# optimazers_list.append(optimizers.Adam(learning_rate=0.01, beta_1=0.1, beta_2=0.5, amsgrad=True))
# baseline_model(optimazers_list, ("Adam(default)", "Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=True)", "Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=True)", "Adam(learning_rate=0.01, beta_1=0.1, beta_2=0.5, amsgrad=True)"))

# Поиск лучшего из лучших
# optimazers_list.append(optimizers.SGD(learning_rate=0.1, momentum=0.8))
# optimazers_list.append(optimizers.Adagrad(learning_rate=0.1))
# optimazers_list.append(optimizers.RMSprop())
# optimazers_list.append(optimizers.Adam(learning_rate=0.01, beta_1=0.1, beta_2=0.5, amsgrad=True))
# baseline_model(optimazers_list, ("SGD(learning_rate=0.1, momentum=0.8)", "Adagrad(learning_rate=0.1)", "RMSProp(default)", "Adam(learning_rate=0.01, beta_1=0.1, beta_2=0.5, amsgrad=True)"))

def read_and_predict(path):
    # Загрузка изображения
    image = Image.open(path).convert('L')
    # Преобразовать изображение в массив numpy
    data = asarray(image)
    data = data.reshape((1, 28, 28))
    Y = model.predict_classes(data)
    print("prediction for file " + path + " -------> " "[ " + np.array2string(Y[0]) + " ]")
    return data

model = baseline_model([optimizers.Adam(learning_rate=0.01, beta_1=0.1, beta_2=0.5, amsgrad=True)], ["Adam"]) #[0.9789]
print("---------------------------------------------------------")
read_and_predict("numeral_2.png")
read_and_predict("numeral_3.png")
read_and_predict("numeral_4.png")
read_and_predict("numeral_8.png")