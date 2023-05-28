import cv2
from simple_facerec import SimpleFacerec

facerecModal = SimpleFacerec()

# training mô hình
facerecModal.load_encoding_images('images/')

name = input("Nhập tên của bạn: ")

# Lấy video từ webcam
cap = cv2.VideoCapture(0)

def save_face(name, frame, faces):
    for (y1, x2, y2, x1) in faces:
        crop = frame[x1-100:x2 + 100, y1-50:y2 + 200]
        cv2.imwrite('images/' + str(name) + '.jpg', crop)
    return

while True:
    # Lấy ra các frame
    ret, frame = cap.read()

    # phát hiện và nhận diện khuôn mặt
    face_locations, face_names = facerecModal.detect_known_faces(frame)

    cv2.imshow('frame', frame)

    faces = face_locations

    # Nhấn Esc để kết thúc chương trình
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('s'):
        save_face(name, frame, faces)

# clear

cap.release()
cv2.destroyAllWindows()
