import os
import cv2

folder_path = "customleafdata"

def process_images_in_folder(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        process_images_in_subfolder(subfolder)

def process_images_in_subfolder(subfolder_path):
    image_files = [f.path for f in os.scandir(subfolder_path) if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
    output_folder = os.path.join("output", os.path.basename(subfolder_path))
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_files:

        output_file = os.path.join(output_folder, f"processed_{os.path.basename(image_path)}")
        filter_non_green_lab(image_path, output_file)

def filter_non_green_lab(image_path, output_path):

    img = cv2.imread(image_path)
    median = cv2.medianBlur(img, 5)
    # cv2.imshow("median", median)
    # cv2.waitKey(0)
    lab_img = cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab_img", lab_img)
    # cv2.waitKey(0)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    # cv2.imshow("original", img)
    # cv2.imshow("L Channel", l_channel)
    # cv2.imshow("a Channel", a_channel)
    # cv2.imshow("b Channel", b_channel)
    # cv2.waitKey(0)

    green_mask_lab = cv2.inRange(a_channel, 128, 255)
    result_non_green_lab = cv2.bitwise_and(img, img, mask=green_mask_lab)

    # cv2.imshow("result_non_green_lab", result_non_green_lab)
    # cv2.imshow("original", img)
    # cv2.waitKey(0)
    cv2.imwrite(output_path, result_non_green_lab)


process_images_in_folder(folder_path)
