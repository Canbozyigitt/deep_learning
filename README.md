## Yapay Sinir Ağları

**Yapay sinir ağları**, insan beyninin çalışma prensibini taklit eden matematiksel modellerdir. Verileri alır, işler ve öğrenerek tahminler yapar. 🚀

Örneğin, bir yapay sinir ağına el yazısı rakamları öğretirsek, yeni bir rakam gösterildiğinde hangi sayı olduğunu tahmin edebilir. 📊

Bunu **bağlantılı "nöronlar"** ile yapar

🟢 **Girişler** → 🎛 **Ağırlıklandırma & Aktivasyon** → 🔵 **Çıkış tahmini**

Yani, sinir ağları **verileri öğrenen ve örüntüler çıkaran algoritmalardır** 😊

## Yapay Sinir  Ağları Modelleri

### **1) Sequential Model (Sıralı Model)**

📌 **Ne Zaman Kullanılır?**

- Katmanlar **sırasıyla** birbirini takip ediyorsa.
- Basit ileri beslemeli (feedforward) ağlar için uygundur

### **2) Functional API (Fonksiyonel Model)**

📌 **Ne Zaman Kullanılır?**

- Birden fazla giriş veya çıkış varsa.
- Katmanlar **birbirine bağlı ama sıralı değilse** (örn. birden fazla yol varsa)

### **3) Model Subclassing (Özel Model Tanımlama)**

📌 **Ne Zaman Kullanılır?**

- **Daha karmaşık yapılar veya özel ileri yayılım işlemleri** gerektiğinde.
- Özel ağırlık güncellemeleri, özel kayıp fonksiyonları kullanmak istenirse.

### **4) Transfer Learning Modelleri (Önceden Eğitilmiş Modeller)**

📌 **Ne Zaman Kullanılır?**

- Büyük veri setlerinde eğitilmiş hazır bir modeli alıp, kendi verimizle eğitmek için.

📌 **Örnek Modeller:**

     ✅ `VGG16`✅ `ResNet50`✅ `MobileNet`


### 📌 **Öğrendiklerin:**

### 1️⃣ **Veri Ön İşleme:**

✅ **CIFAR-10 veri setini yükledin** (60.000 renkli görüntü, 10 sınıf).

✅ **Piksel değerlerini 0-1 arasına normalize ettin** (Böylece model daha hızlı ve stabil öğrenir).

### 2️⃣ **Veri Artırma (Data Augmentation):**

✅ **Görüntülere rastgele dönüşümler ekledin** (Yatay çevirme, döndürme, yakınlaştırma).

✅ **Bu teknik, modelin genelleştirme gücünü artırır ve overfitting’i önler.**

### 3️⃣ **CNN Modeli Tanımlama:**

✅ **3 evrişim (Conv2D) katmanı ekledin** → Görüntüdeki kenarları, dokuları ve özellikleri yakalar.

✅ **Havuzlama (MaxPooling) katmanları ekledin** → Modelin daha verimli öğrenmesini sağladı.

✅ **Tam bağlı (Dense) katmanları ekledin** → Son katmanda 10 sınıfa ayıran `softmax` aktivasyonu kullandın.

### 4️⃣ **Modeli Derleme ve Eğitme:**

✅ **Adam optimizasyon algoritmasını kullandın** → Modelin daha hızlı ve stabil öğrenmesini sağladı.

✅ **Loss fonksiyonu olarak sparse_categorical_crossentropy kullandın** → Çok sınıflı sınıflandırma için uygun.

✅ **Modeli 15 epoch boyunca eğittin** ve validasyon verisi ile değerlendirdin.

### 5️⃣ **Modeli Test Etme:**

✅ **Eğitilen modelin test verisindeki doğruluğunu hesapladın.**

### 6️⃣ **Eğitim ve Validasyon Kayıplarını Çizme:**

✅ **Eğitim ve validasyon kayıplarını çizerek modelin nasıl öğrendiğini görselleştirdin.**

✅ **Eğer validasyon kaybı erken yükseliyorsa, model overfitting yapıyor olabilir!**
### 📌 **Doğal Dil İşlemenin (NLP) Temel Adımları:**

1️⃣ **Önişleme (Preprocessing)** → Metni temizleme (küçük harfe çevirme, noktalama işaretlerini kaldırma vb.)

2️⃣ **Tokenizasyon** → Metni kelimelere veya cümlelere bölme.

3️⃣ **Dizileştirme (Text-to-Sequence)** → Kelimeleri sayısal verilere çevirme.

4️⃣ **Gömme (Embedding)** → Kelimeleri vektörler haline getirme (Word2Vec, GloVe, BERT).

5️⃣ **Model Eğitme** → RNN, LSTM veya Transformer gibi modeller kullanarak metni analiz etme.

6️⃣ **Tahmin ve Değerlendirme** → Modelin çıktısını yorumlama.

---

### 🔍 **RNN ve LSTM Nedir?**

✅ **RNN (Recurrent Neural Networks):** Sıralı veriler (metin, konuşma, zaman serileri) için kullanılan bir sinir ağı türüdür. Geçmiş bilgileri hafızasında tutarak sonraki adımları tahmin eder. Ancak **geri yayılım sırasında bilgi kaybına (vanishing gradient)** neden olabilir.

✅ **LSTM (Long Short-Term Memory):** RNN’in geliştirilmiş versiyonudur. **Uzun vadeli bağımlılıkları öğrenebilir** ve bilgi kaybı problemini çözer. LSTM hücreleri, **hangi bilgiyi saklayacağına ve unutacağına karar veren kapılar içerir**.
