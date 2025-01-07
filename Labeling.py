import cv2

#sciezka do obrazu
img_path = "./zdjecia/zdjecie1.png"
#wczytanie obrazu
img = cv2.imread(img_path)

#przeskalowanie obrazu do wymiarow 1920x1080   
resized_img = cv2.resize(img, (1920, 1080))
    
#wybor obszaru do zaznaczenia
x, y, w, h = cv2.selectROI("Select ROI", resized_img, fromCenter=False, showCrosshair=False)

#wyswietlenie zaznaczonych wspolrzednych
print(f"[{x}, {y}, {x + w}, {y + h}]")
    
#zamkniecie wszystkich okien
cv2.destroyAllWindows()