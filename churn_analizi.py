import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler  # <-- YENİ KAHRAMANIMIZ
from sklearn.ensemble import RandomForestClassifier

# --- 1. VERİYİ YÜKLEME VE TEMİZLEME ---
print("Veri yükleniyor...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"Veri Hazır! Satır sayısı: {len(df)}")
print("-" * 30)

# --- 2. VERİYİ HAZIRLAMA VE ÖLÇEKLENDİRME ---
df.drop('customerID', axis=1, inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Random Forest ölçeklendirmeden etkilenmez, ancak pipeline tutarlılığı için uygulanmıştır.
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# --- 3. MODELLEME ---
print("Model eğitiliyor...")

# Train/Test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Artık veriler ölçekli olduğu için max_iter artırmasak bile hızlıca çözer.
# Ama yine de işimizi sağlama alalım.
# class_weight='balanced': Azınlıktaki 'Gidenleri' daha ciddiye al demektir.
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

tahminler = model.predict(X_test)
accuracy = accuracy_score(y_test, tahminler)

print("-" * 30)
print(f"MODEL SONUCU (Accuracy): %{accuracy * 100:.2f}")
print("-" * 30)
print("Confusion Matrix:")
print(confusion_matrix(y_test, tahminler))

# --- HANGİ ÖZELLİK DAHA ÖNEMLİ? (FEATURE IMPORTANCE) ---
import pandas as pd # Emin olmak için tekrar import (zararı yok)

# X artık bir numpy array olduğu için sütun isimlerini orijinal tablodan (df_encoded) almalıyız.
# Ancak 'Churn' sütununu atmayı unutmamalıyız çünkü X'in içinde Churn yok.
feature_names = df_encoded.drop('Churn', axis=1).columns

# Şimdi feature_names değişkenini kullanalım
onem_sirasi = pd.Series(model.feature_importances_, index=feature_names)

# Sıralama ve çizim kısmı aynı
onem_sirasi = onem_sirasi.sort_values(ascending=False).head(10) 

plt.figure(figsize=(10, 6))
sns.barplot(x=onem_sirasi.values, y=onem_sirasi.index, hue=onem_sirasi.index, palette='viridis', legend=False)
plt.title('Müşteri Terkini Etkileyen En Önemli 10 Faktör')
plt.xlabel('Önem Derecesi')
plt.show()

# --- EŞİK DEĞERİ (THRESHOLD) AYARI ---
# Modelden sadece 0 veya 1 değil, ihtimalleri (olasılıkları) isteyelim
y_pred_proba = model.predict_proba(X_test)[:, 1] # Sadece '1' olma ihtimalini al

# Yeni eşik değerimiz: %30 (0.30)
yeni_esik = 0.30
yeni_tahminler = (y_pred_proba >= yeni_esik).astype(int)

print("-" * 30)
print(f"YENİ EŞİK DEĞERİ ({yeni_esik}) İLE SONUÇLAR:")
print("-" * 30)

# Yeni Confusion Matrix
yeni_cm = confusion_matrix(y_test, yeni_tahminler)
print(yeni_cm)

# Yeni Recall Değeri (Yakalananlar / Toplam Gidenler)
# Matrisin yapısı: [[TN, FP], [FN, TP]] -> TP = yeni_cm[1, 1], FN = yeni_cm[1, 0]
tp = yeni_cm[1, 1]
fn = yeni_cm[1, 0]
yeni_recall = tp / (tp + fn)

print(f"Eski Recall (Yakalananlar): 167 kişi")
print(f"Yeni Recall (Yakalananlar): {tp} kişi")
print(f"Yakalam Oranı (Recall Score): %{yeni_recall*100:.2f}")

# --- KORELASYON HARİTASI ---
plt.figure(figsize=(12, 8))
# Sadece sayısal sütunları seçip korelasyonuna bakalım
# df_encoded kullanıyoruz çünkü her şey sayıya döndü
corr = df_encoded.corr()

# Churn ile en çok ilişkili olanları görelim
sns.heatmap(corr[['Churn']].sort_values(by='Churn', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Değişkenlerin Churn ile Korelasyonu')
plt.show()