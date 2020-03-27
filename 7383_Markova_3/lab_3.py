# Подключение модулей
import numpy as np
from tensorflow.keras.layers import Dense             # Один из типов слоёв в Keras для создания слоя нейронов
from tensorflow.keras.models import Sequential        # Для создания простых моделей используют Sequential,
                                                      # При использовании Sequential достаточно добавить несколько своих слоев
import matplotlib.pyplot as plt                       # Библиотека для визуализации графиков
from tensorflow.keras.datasets import boston_housing

# Загрузка набора данных для Бостона
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Вывод данных для просмотра
print(train_data.shape)                               # 404 обучающих образца с 12 числовыми признаками
print(test_data.shape)                                # 102 контрольных образца с 12 числовыми признаками
print(test_targets)                                   # Цели — медианные значения цен на дома, занимаемые собственниками, в тысячах долларов

# Нормализация данных
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

# Определение (создание) модели
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# mse - среднеквадратичная ошибка
# mae - средняя абсолютная ошибка

# K-fold cross-validation
k = 6
num_val_samples = len(train_data) // k
num_epochs = 50
all_scores = []
mae_histories = []

for i in range(k):
    print('processing fold #', i)
    # Подготовка проверочных данных: данных из блока с номером k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Подготовка обучающих данных: данных из остальных блоков
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_target = np.concatenate([train_targets[: i * num_val_samples],
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)

    # Конструирование модели Keras (уже скомпилированной)
    model = build_model()

    # Обучение модели (в режиме без вывода сообщений, verbose = 0)
    history = model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1,
                        validation_data=(val_data, val_targets))
    mae = history.history['mean_absolute_error']
    v_mae = history.history['val_mean_absolute_error']
    x = range(1, num_epochs + 1)
    mae_histories.append(v_mae)
    plt.figure(i + 1)
    plt.plot(x, mae, 'c', label='Training MAE')
    plt.plot(x, v_mae, 'g', label='Validation MAE')
    plt.title('Absolute error')
    plt.ylabel('Absolute error')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()

# Создание истории последовательных средних оценок проверки  по K блокам
average_mae_history = [np.mean([x[i] for x in mae_histories]) for i in range(num_epochs)]

# Сохранение результатов в файл
plt.figure(0)
plt.plot(range(1, num_epochs + 1), average_mae_history, 'g')
plt.xlabel('Epochs')
plt.ylabel("Mean absolute error")
plt.grid()
figs = [plt.figure(n) for n in plt.get_fignums()]
for i in range(len(figs)):
        figs[i].savefig("./%d.png" %(i), format='png')

#print(all_scores)
#print(np.mean(all_scores))
