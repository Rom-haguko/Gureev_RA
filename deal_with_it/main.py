import cv2
import numpy as np

def pixelate_image_area(input_img, block_dimensions=(5,5)):
    pixelated_img = np.zeros_like(input_img)
    y_increment = pixelated_img.shape[0] // block_dimensions[0]
    x_increment = pixelated_img.shape[1] // block_dimensions[1]

    if y_increment == 0: y_increment = 1
    if x_increment == 0: x_increment = 1

    for current_y in range(0, input_img.shape[0], y_increment):
        for current_x in range(0, input_img.shape[1], x_increment):
            for color_ch_index in range(input_img.shape[2]):
                block = input_img[current_y : current_y + y_increment, current_x : current_x + x_increment, color_ch_index]
                if block.size > 0:
                    mean_val = np.mean(block)
                    pixelated_img[current_y : current_y + y_increment, current_x : current_x + x_increment, color_ch_index] = mean_val
    return pixelated_img

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

video_device = cv2.VideoCapture(0)
video_device.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
video_device.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
video_device.set(cv2.CAP_PROP_EXPOSURE, 1000)

eye_pair_detector = cv2.CascadeClassifier("haarcascade-eye.xml")

overlay_graphic = cv2.imread("deal-with-it.png")


while video_device.isOpened():
    success, live_feed_image = video_device.read()
   

    blurred_feed = cv2.GaussianBlur(live_feed_image, (7, 7), 0)
    grayscale_feed = cv2.cvtColor(blurred_feed, cv2.COLOR_BGR2GRAY)

    eye_center_coordinates = []
    eye_horizontal_boundaries = []

    detected_eye_zones = eye_pair_detector.detectMultiScale(grayscale_feed, scaleFactor=2.6, minNeighbors=20)

    for (eye_x_coord, eye_y_coord, eye_width, eye_height) in detected_eye_zones[:2]:
        eye_center_coordinates.append([eye_x_coord + eye_width / 2, eye_y_coord + eye_height / 2])
        eye_horizontal_boundaries.append(eye_x_coord)
        eye_horizontal_boundaries.append(eye_x_coord + eye_width)

    if len(eye_center_coordinates) == 2:
        glasses_pos_x = int((eye_center_coordinates[0][0] + eye_center_coordinates[1][0]) / 2)
        glasses_pos_y = int((eye_center_coordinates[0][1] + eye_center_coordinates[1][1]) / 2)

        eye_span = max(eye_horizontal_boundaries) - min(eye_horizontal_boundaries)
        if overlay_graphic.shape[1] > 0:
             base_scale = eye_span / overlay_graphic.shape[1]
        else:
             base_scale = 1.0

        overlay_magnification_factor = 1.5
        
        target_w = int(overlay_graphic.shape[1] * base_scale * overlay_magnification_factor)
        target_h = int(overlay_graphic.shape[0] * base_scale * overlay_magnification_factor)
        
        target_w = (target_w // 2) * 2
        target_h = (target_h // 2) * 2

        if target_w > 0 and target_h > 0:
            resized_overlay_graphic = cv2.resize(overlay_graphic, (target_w, target_h))

            mask_lower_threshold = np.array([230, 230, 230])
            mask_upper_threshold = np.array([255, 255, 255])
            background_alpha_mask = cv2.inRange(resized_overlay_graphic, mask_lower_threshold, mask_upper_threshold)
            foreground_alpha_mask = cv2.bitwise_not(background_alpha_mask)

            roi_y_start = glasses_pos_y - target_h // 2
            roi_y_end = glasses_pos_y + target_h // 2
            roi_x_start = glasses_pos_x - target_w // 2
            roi_x_end = glasses_pos_x + target_w // 2

            if (roi_y_start >= 0 and roi_y_end <= live_feed_image.shape[0] and
                    roi_x_start >= 0 and roi_x_end <= live_feed_image.shape[1]):
                
                target_region = live_feed_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                fg_mask_rgb = cv2.cvtColor(foreground_alpha_mask, cv2.COLOR_GRAY2BGR)
                bg_mask_rgb = cv2.cvtColor(background_alpha_mask, cv2.COLOR_GRAY2BGR)
                
                overlay_foreground = cv2.bitwise_and(resized_overlay_graphic, fg_mask_rgb)
                frame_background_part = cv2.bitwise_and(target_region, bg_mask_rgb)

                combined_roi = cv2.add(frame_background_part, overlay_foreground)
                live_feed_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = combined_roi

    key_input_code = cv2.waitKey(1)
    if key_input_code != -1:
        pressed_char = chr(key_input_code & 0xFF)
        if pressed_char == "q":
            break

    cv2.imshow("Camera", live_feed_image)

video_device.release()
cv2.destroyAllWindows()
