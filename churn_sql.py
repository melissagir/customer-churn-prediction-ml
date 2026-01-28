import pandas as pd
import sqlite3

# 1. Veriyi yine internetten (veya kaydettiğin CSV'den) çekelim
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Temizlik (SQL sayısal işlemlerde hata vermesin diye)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 2. SANAL VERİTABANI YARATMA (Sihir Burada!)
# RAM üzerinde geçici bir veritabanı kuruyoruz
conn = sqlite3.connect(':memory:') 

# Pandas dataframe'ini SQL tablosuna çevirip içine atıyoruz
# Tablomuzun adı artık: 'musteriler'
df.to_sql('musteriler', conn, index=False, if_exists='replace')

# 3. SQL SORGUSU YAZMA FONKSİYONU
def sql_calistir(sorgu):
    # SQL sorgusunu çalıştır ve sonucu yine tablo olarak göster
    return pd.read_sql(sorgu, conn)

print("SQL Veritabanı Hazır! Tablo adı: 'musteriler'")

print("\n--- Soru 1: İlk 5 Müşteriyi Getir ---")
query1 = "SELECT * FROM musteriler LIMIT 5"
print(sql_calistir(query1))

print("\n--- Soru 2: Yüksek Faturalı Kadın Müşteriler ---")
query2 = """
SELECT gender, MonthlyCharges, Churn 
FROM musteriler 
WHERE gender = 'Female' AND MonthlyCharges > 100
LIMIT 5
"""
print(sql_calistir(query2))

print("\n--- Soru 3: Sözleşme Türüne Göre Ortalama Fatura ---")
query3 = """
SELECT Contract, AVG(MonthlyCharges) as Ortalama_Fatura
FROM musteriler
GROUP BY Contract
ORDER BY Ortalama_Fatura DESC
"""
print(sql_calistir(query3))

print("\n--- Soru 4:Ödeme yöntemi (PaymentMethod) 'Electronic check' olan kaç kişi var? ---")
query4 = """
SELECT COUNT(*) 
FROM musteriler 
WHERE PaymentMethod = 'Electronic check'
"""
print(sql_calistir(query4))

print("\n--- Soru 5: Hangi İnternet Servisi Daha Çok Kazandırıyor? ---")

query5 = """
SELECT InternetService, SUM(TotalCharges) as Toplam_Ciro
FROM musteriler
GROUP BY InternetService
ORDER BY Toplam_Ciro DESC
"""

print(sql_calistir(query5))

print("\n--- Soru 6: Churn Oranı %25'ten Yüksek Olan Ödeme Yöntemleri (HAVING) ---")

query6 = """
SELECT PaymentMethod, AVG(Churn) as Kayip_Orani
FROM musteriler
GROUP BY PaymentMethod
HAVING Kayip_Orani > 0.25
ORDER BY Kayip_Orani DESC
"""

print(sql_calistir(query6))