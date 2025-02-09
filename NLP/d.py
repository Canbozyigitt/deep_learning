import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
import numpy as np

# 1️⃣ IMDb Veri Setini Yükleme
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# 2️⃣ Verileri Aynı Uzunluğa Getirme (Padding)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# 3️⃣ Modeli Tanımlama
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=200),  # Kelime vektörleri
    layers.LSTM(64, return_sequences=True),  # İlk LSTM katmanı
    layers.LSTM(64),  # İkinci LSTM katmanı
    layers.Dense(1, activation='sigmoid')  # Çıkış katmanı (Pozitif/Negatif)
])

# 4️⃣ Modelin Derlenmesi
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5️⃣ Modelin Eğitilmesi
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 6️⃣ Modelin Test Edilmesi
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest doğruluğu: {test_acc:.4f}')
