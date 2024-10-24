
ÖDEV 3

A. Uygulama Adımları

1. Gerekli Kütüphanelerin Yüklenmesi
Derste kullanılan kütüphanelerin hepsini çalışma ortamınıza yükleyin,
ihtiyaç halinde farklı kütüphaneleri de yükleyebilirsiniz.

pip install ultralytics komutu  ile kurdum
from ultralytics import YOLO modelimi yükledim
Bu satır, YOLO nesne tespit modeliyle çalışabilmemiz için gerekli olan YOLO sınıfını projemize dahil ettik.
model = YOLO('yolov11n.pt') modelimi seçtim.
yolo11x.pt: En büyük model 
yolo11m.pt: Orta boy model 
yolo11l.pt: Büyük model

2. Fotoğrafların Çekilmesi ve Videonun Kaydedilmesi ve Yüklenmesi
Kendi kameranızla farklı senaryolara sahip 10 adet fotoğraf çekin ve 1
adet video kaydedin. Bu dosyaları çalışma ortamınıza dahil edin.
10 Adet kamera görüntümü ve 1 adet kamera videomu yükledim

3. Model Seçimi
Yolov11 modelinin farklı versiyonlarını (m, l, x, vb.) 
Aşağıdaki modelleri  seçip, ortamınıza yükledim.
yolo11x.pt: En büyük model 
yolo11m.pt: Orta boy model 
yolo11l.pt: Büyük model
Model2 olarak yolo11m.pt  Yükledim.

4. Modelin Uygulanması
Seçtiğiniz model üzerinden kendi çektiğiniz verileri çalıştırın.

Kendi belirlediğim Screnhost_1.jpg,……..Screnhost_10.jpg Resim dosyası ve video_1.mp4 video dosyasını çalıştırdım

5. Sonuçları Görselleştirme
Elde ettiğiniz sonuçları gözden geçirin. Hangi nesneler tespit edildi?
Hangi nesneler atlandı? Tespit ettiğiniz sonuçları grafikler veya tablolar
ile sunarak analiz edin ve daha sonra bu grafikleri ve tabloları bir cümle
ile yorumlayın.

Gözden geçirdiğim de   video ve resimlerdeki tüm   nesneleri (insan, eşya, araç, tır, otobüs vs.) tespit edebildim
Orta boy YOLO modeli (mode2) kullanılarak görüntü üzerinde nesne tespiti gerçekleştirdim.
conf=0.15:  Parametresi olduğunda, model %15 veya daha fazla güvendiği nesneleri tespit edecektir. Bu değer ne kadar yüksek olursa daha az yanlış nesneler tespit edecektir

6. Rapor Hazırlığı
Fotoğraflarınız üzerinde yapılan nesne tespit sonuçlarını rapor halinde düzenleyin.

videos = ["C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\videos\\video_1.mp4"] for video_path in videos: results = mode2.predict(video_path, stream=True, conf=0.15, show=True, save=True, device='mps') print(results)]
print(results)

Raporunuzda şunlar yer almalı:
● Kullandığınız modelin adı ve sürümü
Modelin adi : ultralytics Sürümü :Yolo11 

● Tespit edilen nesneler ve başarı oranları
Gözden geçirdiğim de   video ve resimlerdeki tüm   nesneleri (insan, eşya, araç, tır, otobüs vs.) tespit edebildim başarı oranları  %15 olmasına rağmen  tüm nesneleri gözlemledim.

● Yanlış tespitler veya gözden kaçan nesneler
Yok
● Çıktılara yönelik kişisel yorumlar ve analizler

Nesnelerin Türkçe   olarak tanımlaması yaptırmak istiyorum. Nesneleri ekrana zom yaptırmak daha anlaşılır olur düşünüyorum.



