import cv2
from simple_facerec import SimpleFacerec

str = SimpleFacerec()

# training mô hình
str.load_encoding_images('images/')

# Lấy video từ webcam
cap = cv2.VideoCapture(0)

while True:
    # Lấy ra các frame
    ret, frame = cap.read()

    # phát hiện và nhận diện khuôn mặt
    face_locations, face_names = str.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):

        # Lấy ra tọa độ khuôn mặt
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Vẽ tên lên các frame
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        # Vẽ một hình chữ nhật với tọa độ đã lấy được ở trên
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow('frame', frame)

    # Nhấn Esc để kết thúc chương trình
    key = cv2.waitKey(1)
    if key == 27:
        break

# clear

cap.release()
cv2.destroyAllWindows()
