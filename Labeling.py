import cv2

img_path = "./zdjecia/zdjecie5.png"
img = cv2.imread(img_path)
    
resized_img = cv2.resize(img, (1920, 1080))
    
x, y, w, h = cv2.selectROI("Select ROI", resized_img, fromCenter=False, showCrosshair=False)

print(f"[{x}, {y}, {x + w}, {y + h}]")
    
cv2.destroyAllWindows()