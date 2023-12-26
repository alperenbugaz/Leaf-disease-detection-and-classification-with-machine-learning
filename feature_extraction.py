import math
import cv2
import numpy as np
import os
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from math import sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from numpy import asarray


def process_images_in_folder(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    data = []
    for subfolder in subfolders:
        process_images_in_subfolder(subfolder, data)
    column_names = ["Image","Energy", "Homogeneity", "RMSE", "Smoothness", "Skewness", "Kurtosis",
                    "Contrast", "SD", "Mean", "Entropy", "Correlation", "Inverse Difference Moment", "Variance","Label"]
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv("image_features.csv", index=False)
    df.to_excel("image_features.xlsx", index=False)

def process_images_in_subfolder(subfolder_path, data):
    image_files = [f.path for f in os.scandir(subfolder_path) if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label = os.path.basename(subfolder_path)
    for image_path in image_files:
        extract_image_features(image_path, label, data)

def shannon_entropy(image):
    _, counts = np.unique(image, return_counts=True)
    probabilities = counts / np.sum(counts)
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_value

def extract_image_features(image_path, label, data):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü gri tonlamalı olarak yükle
    _, thresholded_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #-----------------------------------------
    energy = np.sum(thresholded_img ** 2)
    homogeneity = cv2.HuMoments(cv2.moments(thresholded_img)).flatten()[0]
    rmse = sqrt(np.mean((thresholded_img - np.mean(thresholded_img)) ** 2))
    smoothness = 1 - (1 / (1 + np.var(thresholded_img)))
    skewness = skew(thresholded_img.flatten())
    kurt = kurtosis(thresholded_img.flatten())
    variance = np.var(thresholded_img)
    hist = cv2.calcHist([thresholded_img], [0], None, [256], [0, 256])
    hist_normalized = hist / np.sum(hist)
    contrast = np.sum(hist_normalized * (np.arange(256) - np.mean(np.arange(256))) ** 2)
    sd = np.std(thresholded_img)
    mean = np.mean(thresholded_img)
    entropy_value = shannon_entropy(thresholded_img)
    correlation = np.correlate(thresholded_img.flatten(), thresholded_img.flatten())[0]
    inverse_difference_moment = np.sum(1 / (1 + (np.arange(256) - np.arange(256)[:, np.newaxis]) ** 2))

    image_features = [os.path.basename(image_path), energy, homogeneity, rmse, smoothness, skewness, kurt,
                      contrast, sd, mean, entropy_value, correlation, inverse_difference_moment, variance, label]
    image_features = [0 if (isinstance(feature, (int, float)) and math.isnan(feature)) else feature for feature in
                      image_features]

    data.append(image_features)


def normalize(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.drop('Image', axis=1)
    selected_columns = ["Energy", "Homogeneity", "RMSE", "Smoothness", "Skewness", "Kurtosis",
                        "Contrast", "SD", "Mean", "Entropy", "Correlation", "Inverse Difference Moment", "Variance"]
    df_selected = df[selected_columns]
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)
    df_normalized = pd.concat([df_normalized, df[['Label']]], axis=1)
    df_normalized.to_csv(output_file, index=False)
    df_normalized.to_excel("normalized.xlsx", index=False)

if __name__ == "__main__":
    main_folder_path = "output"
    process_images_in_folder(main_folder_path)
    normalize("image_features.csv", "normalized.csv")
