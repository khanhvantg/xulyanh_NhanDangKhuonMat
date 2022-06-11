# Nguyễn Đặng Khánh Văn - 19110312
# Hoàng Quốc Việt       - 19110315
# Nguồn tham khảo: Code thầy gửi trên trang dạy học số
import cv2
import time
import os
import pathlib

# Load tất cả các file trong folder video vào mảng videos
videos=os.listdir('video')

def getImageFromVideo(name,fileName):
    cap = cv2.VideoCapture('video/'+name)
    # Resolution 640*480
    time.sleep(1)
    if cap is None or not cap.isOpened():
        print('Khong the mo file video')
        return
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    n = 1
    dem = 0
    p = pathlib.Path('image/'+fileName)
    p.mkdir(exist_ok=True )
    while True:
        [success, img] = cap.read()
        ch = cv2.waitKey(30)
        if success:
            #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            imgROI = img[40:(40+480), :] 

            imgROI = cv2.resize(imgROI, (250, 270))
            cv2.imshow('Image', imgROI)
        else:
            break 
        if n % 4 == 0:
            filename = 'image/'+fileName+'/'+fileName+'_%04d.bmp' % (dem)
            cv2.imwrite(filename, imgROI)
            dem = dem + 1
        n = n + 1
    return


if __name__ == "__main__":
    for index in range(len(videos)):
        # Tách tên file [0] và đuôi file [1]
        filename=os.path.splitext(videos[index])
        getImageFromVideo(videos[index],filename[0])
