# Подключение модулей
import numpy as np                                    # Общие математические и числовые операции
from var4 import gen_sequence                         # Код с генерацией последовательности
import matplotlib.pyplot as plt                       # Библиотека для визуализации графиков
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def gen_data_from_sequence(seq_len=1000, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return (past, future)


data, res = gen_data_from_sequence(seq_len=1500, lookback=10)

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]

# Создание модели
model = Sequential()
model.add(layers.GRU(64, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(64, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.25))
model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.25))
model.add(layers.Dense(8))
model.add(layers.Dense(1))

# Инициализация параметров обучения
model.compile(optimizer='nadam', loss='mse')

# Обучение сети
history = model.fit(train_data, train_res, epochs=20, validation_data=(val_data, val_res))

# Построение графика ошибки
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss, 'g', label='Train')
plt.plot(range(len(val_loss)), val_loss, 'c', label='Validation')
plt.grid()
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss.png', format='png', dpi=240)
plt.clf()

# Построение графика последовательности
predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length, predicted_res, 'g', label='Predicted')
plt.plot(pred_length, test_res, 'c', label='Generated')
plt.title('Sequence')
plt.xlabel('x')
plt.ylabel('Sequence')
plt.grid()
plt.legend()
plt.savefig('Sequence.png', format='png', dpi=240)