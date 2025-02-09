import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. MNIST Veri Setini Yükleme
# El yazısı rakamlarından oluşan MNIST veri setini yükledik.
dataset = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# 2. Normalizasyon (0-255 -> 0-1 arası değerler)
# Piksel değerlerini 0 ile 1 arasına ölçeklendirdik, böylece model daha iyi öğrenir.
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Modelin Tanımlanması
# Yapay sinir ağı modelimizi oluşturduk.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 28x28 boyutundaki görüntüleri tek boyutlu vektöre çeviriyoruz.
    keras.layers.Dense(128, activation='relu'),  # 128 nöronlu gizli katman ekledik ve ReLU aktivasyon fonksiyonu kullandık.
    keras.layers.Dense(10, activation='softmax') # Çıkış katmanı: 10 sınıf (0-9 rakamları) için softmax aktivasyonu kullanıyoruz.
])

# 4. Modelin Derlenmesi
# Modeli eğitmek için gerekli parametreleri belirledik.
model.compile(optimizer='adam',  # Adam optimizasyon algoritmasını kullandık.
              loss='sparse_categorical_crossentropy',  # Kayıp fonksiyonu olarak uygun bir sınıflandırma yöntemi seçtik.
              metrics=['accuracy'])  # Modelin doğruluğunu ölçmek için accuracy metriğini ekledik.

# 5. Modelin Eğitilmesi
# Modeli 5 epoch boyunca eğittik.
model.fit(x_train, y_train, epochs=5)

# 6. Modelin Test Edilmesi
# Modelimizi test verisi üzerinde değerlendiriyoruz.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest doğruluğu: {test_acc:.4f}')  # Modelin test doğruluğunu ekrana yazdırıyoruz.
