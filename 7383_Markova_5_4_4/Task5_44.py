# Подключение модулей
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential

# Параметры данных
DIM_DATASET = 6
SIZE_TRAINING = 320
SIZE_TEST = 80

# Генерирование данных
def generation_of_dataset(size_of_dataset):
    dataset = np.zeros((size_of_dataset, DIM_DATASET))
    dataset_y = np.zeros(size_of_dataset)
    for i in range(size_of_dataset):
        X = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        dataset[i, :] = (np.round(np.cos(X) + e), np.round(-X + e), np.round(np.sin(X) * X + e), np.round(X ** 2 + e), np.round(-np.fabs(X) + 4), np.round(X - (X ** 2) / 5 + e))
        dataset_y[i] = np.round(np.sqrt(np.fabs(X)) + e)
    return np.round(np.array(dataset), decimals=3), np.array(dataset_y)

# Создание модели
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model

def create_objects():
    # Encoder
    main_input = Input(shape=(DIM_DATASET,), name='main_input')
    encoded = Dense(64, activation='relu')(main_input)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(DIM_DATASET, activation='linear')(encoded)

    # Decoder
    input_encoded = Input(shape=(DIM_DATASET,), name='input_encoded')
    decoded = Dense(32, activation='relu', kernel_initializer='normal')(input_encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(DIM_DATASET, name="out_aux")(decoded)

    # Regression
    predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
    predicted = Dense(32, activation='relu')(predicted)
    predicted = Dense(16, activation='relu')(predicted)
    predicted = Dense(1, name="out_main")(predicted)

    # Использование моделей
    encoded = Model(main_input, encoded, name="encoder")
    decoded = Model(input_encoded, decoded, name="decoder")
    predicted = Model(main_input, predicted, name="regr")

    return encoded, decoded, predicted, main_input

x_train, y_train = generation_of_dataset(SIZE_TRAINING)
x_test, y_test = generation_of_dataset(SIZE_TEST)

# Нормализация данных
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std

encoded, decoded, full_model, main_input = create_objects()

keras_model = build_model()

# Инициализация параметров обучения
keras_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Обучение сети
History = keras_model.fit(x_train, y_train, epochs=40, batch_size=5, verbose=1, validation_data=(x_test, y_test))

# Построение графика ошибки
# loss = History.history['loss']
# v_loss = History.history['val_loss']
# plt.plot(range(1, 41), loss, 'c', label='Train')
# plt.plot(range(1, 41), v_loss, 'g', label='Test')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend()
# plt.grid()
# plt.show()
# plt.clf()

# Регрессионная модель
full_model.compile(optimizer="adam", loss="mse", metrics=['mae'])

# Обучение сети
History = full_model.fit(x_train, y_train, epochs=40, batch_size=5, verbose=1, validation_data=(x_test, y_test))

encoded_data = encoded.predict(x_test)
decoded_data = decoded.predict(encoded_data)

regr = full_model.predict(x_test)
# Построение графика ошибки
# loss = History.history['loss']
# v_loss = History.history['val_loss']
# x = range(1, 41)
#
# plt.plot(x, loss, 'c', label='Train')
# plt.plot(x, v_loss, 'g', label='Test')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend()
# plt.grid()
# plt.show()
# plt.clf()

# Запись в CSV необходимых данных
pd.DataFrame(np.round(regr, 3)).to_csv("result.csv")
pd.DataFrame(np.round(x_test, 3)).to_csv("x_test.csv")
pd.DataFrame(np.round(y_test, 3)).to_csv("y_test.csv")
pd.DataFrame(np.round(x_train, 3)).to_csv("x_train.csv")
pd.DataFrame(np.round(y_train, 3)).to_csv("y_train.csv")
pd.DataFrame(np.round(encoded_data, 3)).to_csv("encoded.csv")
pd.DataFrame(np.round(decoded_data, 3)).to_csv("decoded.csv")

# Сохранение модели
decoded.save('decoder.h5')
encoded.save('encoder.h5')
full_model.save('full.h5')