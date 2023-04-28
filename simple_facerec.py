import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        # Thay đổi kích thước khung để có tốc độ nhanh hơn
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):

        # Tải hình ảnh
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Lưu trữ mã hóa hình ảnh và tên
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Chỉ lấy tên tệp từ đường dẫn tệp ban đầu.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Nhận mã hóa
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Lưu trữ tên tệp và mã hóa tệp
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Tìm tất cả các khuôn mặt và mã hóa khuôn mặt trong khung hình hiện tại của video
        # Chuyển đổi hình ảnh từ màu BGR (mà OpenCV sử dụng) sang màu RGB (mà face_recognition sử dụng)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Xem khuôn mặt có khớp với (các) khuôn mặt đã biết không
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Hoặc thay vào đó, sử dụng khuôn mặt đã biết với khoảng cách nhỏ nhất đến khuôn mặt mới
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Chuyển đổi sang mảng numpy để điều chỉnh tọa độ với thay đổi kích thước khung hình một cách nhanh chóng
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
