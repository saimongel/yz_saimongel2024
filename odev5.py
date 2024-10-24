import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# TESLA  Hisse senedi 
symbol = 'TSLA'

# Son 1 yıllık veriyi alalım
pd = yf.download(symbol, period="1y")

# DataFrame'i görüntüleme
print(pd.head())

# Sadece kapanış fiyatlarını alalım
pd['Close']       

# 20 günlük Basit Hareketli Ortalama (SMA) hesaplama
pd['SMA_20'] = pd['Close'].rolling(window=20).mean()

# 50 günlük Üstel Hareketli Ortalama (EMA) hesaplama
pd['EMA_50'] = pd['Close'].ewm(span=50, adjust=False).mean()

# Kapanış fiyatları, SMA ve EMA'yı görselleştirme
plt.figure(figsize=(14,7))
plt.plot(pd['Close'], label='Kapaniş Fiyati', color='blue')
plt.plot(pd['SMA_20'], label='20 Günlük SMA', color='red', linestyle='--')
plt.plot(pd['EMA_50'], label='50 Günlük EMA', color='green', linestyle='--')
plt.title(f'{symbol} - Kapaniş Fiyatlari ve Hareketli Ortalamalar')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()