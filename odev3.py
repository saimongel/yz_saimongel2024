from ultralytics import YOLO
model = YOLO(f'yolo11x.pt') 
mode2 = YOLO(f'yolo11m.pt')
mode3 = YOLO(f'yolo11l.pt')

images = [
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_1.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_2.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_3.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_4.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_5.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_6.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_7.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_8.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_9.jpg",
    "C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\images\\Screenshot_10.jpg"
] 
for image_path in images:
    results = mode2.predict(image_path, stream=False, conf=0.15, show=True, save=True, device='mps')
print(results)
     
videos = ["C:\\Users\\saim\\Desktop\\TEST\\KURS\\ODEV3\\videos\\video_1.mp4"]
for video_path in videos:
    results = mode2.predict(video_path, stream=False, conf=0.15, show=True, save=True, device='mps')
print(results)


