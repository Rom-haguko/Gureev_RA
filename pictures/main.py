import cv2

template_image_path = "Gureev_picture.png"
video_file_path = "output.avi"

orb_features = 500
knn_match_ratio = 0.75
min_matches_threshold = 15

print("Начата работа программы...")

img_template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=orb_features)

keypoints_template, descriptors_template = orb.detectAndCompute(img_template, None)

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

video = cv2.VideoCapture(video_file_path)

frame_count = 0
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

    if descriptors_frame is not None:
        matches = bf_matcher.knnMatch(descriptors_template, descriptors_frame, k=2)
        good_matches = []
        if matches:
             good_matches = [m for m, n in matches if len((m, n)) == 2 and m.distance < knn_match_ratio * n.distance]

        if len(good_matches) > min_matches_threshold:
            frame_count += 1

video.release()

print(f"Обработано кадров всего: {total_frames}")
print(f"Найдено изображений: {frame_count}")
