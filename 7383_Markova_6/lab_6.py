# Подключение модулей
import numpy as np                                    # Общие математические и числовые операции
import matplotlib.pyplot as plt                       # Библиотека для визуализации графиков
from keras.datasets import imdb                       # Internet Movie Database
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras import Sequential, regularizers

# Загрузка набора данных IMDb, который содержит 50 000 отзывов к кинолентам
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)

data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

# Изучение датасета
print("--------------------------------")
print("Категории:", np.unique(targets))
print("Количество уникальных слов:", len(np.unique(np.hstack(data))))
length = [len(i) for i in data]
print("Средний размер обзора:", np.mean(length))
print("Стандартное отклонение:", round(np.std(length)))
print("--------------------------------")
print("Настроение обзора:", targets[8])
print(data[8])
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[8]])
print(decoded)
print("--------------------------------")

# Подготовка данных
def vectorize(sequences, dimension=10000):
    # Каждый элемент входных данных нашей нейронной сети
    # должен иметь одинаковый размер
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

reviews = [
    "I believe all the hype. Everything that every person said...yes, this show is AMAZING, WONDERFUL.",
    "The best movie I've ever seen. Great storyline and great acting.",
    "A awful and boring movie that is impossible to watch.",
    "It really is horribly inert, and every time Downey opens his mouth to say something unintelligible, the film dies a bit more. I do not advise watching this film, it is disgusting.",
    "Dolittle is great fun for all the family, full of wonder and adventure, and it brings with it a great message of friendship against the odds."
]

review_mood = [1.0, 1.0, 0.0, 0.0, 1.0]

def text_load():
    dictionary = dict(imdb.get_word_index())
    test_x = []
    for string in reviews:
        words = string.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('\n', ' ').split()
        num_words = []
        for word in words:
            word = word.lower()
            word = dictionary.get(word)
            num_words.append(word)
        test_x.append(num_words)
    #print(test_x)
    return test_x

    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()])
    decoded = " ".join([reverse_index.get(i, "#") for i in test_x[1]])
    print(decoded)

reviews = text_load()
for j, i in enumerate(reviews):
    for index, value in enumerate(i):
        if value is None or value > 10000:
            reviews[j][index] = 0

# Векторизация обзоров
data = vectorize(data)
targets = np.array(targets).astype("float32")
review_mood = np.asarray(review_mood).astype("float32")

# Разделим датасет на обучающий и тестировочный наборы
test_x = data[:10000]                                 # \ тестировочный набор
test_y = targets[:10000]                              # / состоит из 10 000

train_x = data[10000:]                                # \ обучающий набор
train_y = targets[10000:]                             # / состоит из 40 000

# Создание модели
model = Sequential()
model.add(Dense(50, activation="relu", input_shape=(10000,)))

model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(50, activation="linear", kernel_regularizer=regularizers.l2()))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(100, activation="relu", kernel_regularizer=regularizers.l2()))
model.add(Dense(1, activation="sigmoid"))

# Инициализация параметров обучения
model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Обучение сети
hist = model.fit(train_x, train_y, batch_size=500, epochs=2, validation_data=(test_x, test_y))

# Построение графика ошибки
history_dict = hist.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.figure(1, figsize=(8, 5))
plt.plot(loss_values, 'c', label='Train')
plt.plot(val_loss_values, 'g', label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.show()

# Построение графика точности
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(acc_values, 'c', label='Train')
plt.plot(val_acc_values, 'g', label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

result = model.evaluate(test_x, test_y)
print("-----------------------------------------------------------")
print('Loss & Accuracy: ', result)

reviews = vectorize(reviews)

custom_loss, custom_acc = model.evaluate(reviews, review_mood)
print('Accuracy:', custom_acc)
prediction = model.predict(reviews)
plt.figure(3, figsize=(8, 5))
plt.title("Dataset predications")
plt.scatter([1, 2, 3, 4, 5], review_mood, marker='*', c='c', s=100, label='Truth')
plt.scatter([1, 2, 3, 4, 5], prediction, c='g', s=50, label='Prediction')
plt.xlabel('Revocation number')
plt.ylabel('Mood')
plt.legend()
plt.grid()
plt.show()

print("--------------------------------")
print(prediction)