import tensorflow as tf

X = tf.constant(
    [[25], [50], [75], [100], [125], [150], [175], [200], [225], [250]],
    dtype=tf.float32,
)  # Количество анекдотов про Штирлица, которые знает человек
Y = tf.constant(
    [[100], [200], [300], [400], [500], [600], [700], [800], [900], [1000]],
    dtype=tf.float32,
)  # Количество людей, которые будут считать рассказчика придурком

# Среднее и стандартное отклонение на тренировочных данных
X_mean = tf.reduce_mean(X)
X_std = tf.math.reduce_std(X)
Y_mean = tf.reduce_mean(Y)
Y_std = tf.math.reduce_std(Y)

# Стандартизация данных
X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std

# Модель линейной регрессии
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])

# Компиляция модели с уменьшенной скоростью обучения
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss="mean_squared_error"
)

# Тренировка модели (с логированием)
model.fit(X, Y, epochs=1000, verbose=1)

# Прогноз для количества людей, которых ты можешь разочаровать
jokes = tf.constant([[170]], dtype=tf.float32)

# Стандартизация ввода на основе статистики тренировочных данных
jokes_standardized = (jokes - X_mean) / X_std

disapointed_people_standardized = model.predict(jokes_standardized)

# Дестандартизация результата (обратное преобразование)
disapointed_people = disapointed_people_standardized * Y_std + Y_mean

print(f"Количество людей, которое ты можешь разочаровать: {disapointed_people[0][0]}")
