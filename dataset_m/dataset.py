import os
import cv2
import numpy as np

def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=70, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_background = np.zeros_like(gray)
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    kernel = np.ones((10, 10), np.uint8)
    black_background = cv2.morphologyEx(black_background, cv2.MORPH_CLOSE, kernel)
    result = img.copy()
    result[black_background == 0] = 0
    return result

def resize_and_preserve_aspect_ratio(image, target_size=(256, 256)):
    height, width, _ = image.shape
    scale_x = target_size[0] / width
    scale_y = target_size[1] / height
    scale_factor = min(scale_x, scale_y)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_img = cv2.resize(image, (new_width, new_height))
    black_background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    offset_x = (target_size[0] - new_width) // 2
    offset_y = (target_size[1] - new_height) // 2
    black_background[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_img
    return black_background

def process_images_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    image_files = [file for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for i, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, image_file)
        processed_image = process_image(image_path)
        resized_image = resize_and_preserve_aspect_ratio(processed_image, target_size=(256, 256))
        resized_image_path = f'resized_{i}.jpg'
        cv2.imwrite(resized_image_path, resized_image)

        print(f"Resized image {i}: {resized_image_path}")


photo_folder_path = 'photos'
process_images_in_folder(photo_folder_path)
