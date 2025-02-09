import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers

import matplotlib.pyplot as plt

# 1️⃣ CIFAR-10 Veri Setini Yükleme
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 2️⃣ Normalizasyon (0-255 -> 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3️⃣ Veri Artırma (Data Augmentation)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),      # Görüntüleri yatay çevirme
    layers.RandomRotation(0.1),           # %10 döndürme
    layers.RandomZoom(0.1),               # %10 yakınlaştırma
])

# 4️⃣ CNN Modeli Tanımlama
model = keras.Sequential([
    data_augmentation,                      # Veri artırma katmanı
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),  
    layers.MaxPooling2D(2,2), 
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2), 
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2), 
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(10, activation='softmax')  # CIFAR-10'da 10 sınıf var
])

# 5️⃣ Modelin Derlenmesi
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6️⃣ Modelin Eğitilmesi
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# 7️⃣ Modelin Test Edilmesi
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest doğruluğu: {test_acc:.4f}')

# 8️⃣ Eğitim ve Validasyon Kaybını Çizme
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Validasyon Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()
