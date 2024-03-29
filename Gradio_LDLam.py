import gradio as gr
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import median_filter
from skimage.filters import threshold_multiotsu, threshold_otsu

def basic_preprocessing(input_image, kernel_size):
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # average filter
    blurred_image = cv2.blur(gray_image, (kernel_size, kernel_size))

    # median filter
    median_filtered_image = median_filter(blurred_image, size=kernel_size)

    # histogram equalizing
    hist_equ_img = cv2.equalizeHist(gray_image)

    return blurred_image, median_filtered_image, hist_equ_img

def calculate_psnr(input_image):
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Calculate MSE
    mse = np.mean((gray_image.astype(np.float64) / 255) ** 2)
    if mse == 0:
        return "Same Image"

    # Calculate PSNR
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)

    return psnr

def kapur_segment(input_image):
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Normalize to 0 - 255
    image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply Kapur's method
    thresholds = threshold_multiotsu(image)

    # Apply the thresholds to get a segmented image
    segmented_image = np.digitize(image, bins=thresholds)

    return segmented_image

def otsu_segment(input_image):
    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Normalize to 0 - 255
    image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply Otsu's thresholding
    thresh = threshold_otsu(image)
    binary = image > thresh

    return binary

def calculate_metrics(pred, mask):
    # normalize eh_mask [0, 255] -> [0, 1]
    norm_mask = np.where(mask, 1, 0)
    # convert 3 channel to 1 channel
    norm_mask_uint8 = norm_mask.astype(np.uint8)
    mask = cv2.cvtColor(norm_mask_uint8, cv2.COLOR_RGB2GRAY)

    # True positive
    tp = np.sum((pred == 1) & (mask == 1))

    # True negative
    tn = np.sum((pred == 0) & (mask == 0))

    # False positive
    fp = np.sum((pred == 1) & (mask == 0))

    # False negative
    fn = np.sum((pred == 0) & (mask == 1))

    # Sensitivity (also known as recall or true positive rate)
    sensitivity = tp / (tp + fn)

    # Specificity (also known as true negative rate)
    specificity = tn / (tn + fp)

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return sensitivity, specificity, accuracy

def image_processing(img, original_mask, kernel_sz):
    # basic img information
    info = img.shape

    # basic preprocessing
    average_img, median_img, hist_equ_img = basic_preprocessing(img, kernel_sz)

    # psnr value
    psnr = calculate_psnr(img)

    # segmentation Kapur method
    kapur_seg_mask = kapur_segment(img)
    kapur_img = ((kapur_seg_mask / kapur_seg_mask.max()) * 255).astype(np.uint8)

    # segmentation otsu method
    otsu_seg_mask = otsu_segment(img)
    otsu_img = cv2.cvtColor((otsu_seg_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Kapur segmentation evaluation
    kapur_sensitivity, kapur_speciality, kapur_accuracy = calculate_metrics(kapur_seg_mask, original_mask)
    metric_kapur = f'Sensitivity: {kapur_sensitivity:.2f} \nSpecificity: {kapur_speciality:.2f} \nAccurarcy: {kapur_accuracy:.2f}'

    # Otsu segmentation evaluation
    otsu_sensitivity, otsu_speciality, otsu_accuracy = calculate_metrics(otsu_seg_mask, original_mask)
    metric_otsu = f'Sensitivity: {otsu_sensitivity:.2f} \nSpecificity: {otsu_speciality:.2f} \nAccurarcy: {otsu_accuracy:.2f}'

    return info, average_img, median_img, hist_equ_img, psnr, kapur_img, metric_kapur, otsu_img, metric_otsu

inputs = [
    gr.Image(type='numpy', label="Input original image"),
    gr.Image(type='numpy', label="Input original mask"),
    gr.Number(label='Kernel size (for average and median filter only)', value=3),
]
outputs = [
    gr.Textbox(label="Shape"),  # img info
    gr.Image(label="Average filtering"),  # avg filtering
    gr.Image(label="Median filtering"),  # median filtering
    gr.Image(label="Histogram equalized img"),  # histogram equalizing
    gr.Number(label="PSNR value"),  # psnr value
    gr.Image(label="Kapur segmentation"),  # segmentation Kapur method
    gr.Textbox(label="Evaluation metrics (Kapur)"),  # Kapur metric
    gr.Image(label="Otsu segmentation"),  # segmentation Otsu method
    gr.Textbox(label="Evaluation metrics (Otsu)"),  # Otsu metrics
]

iface = gr.Interface(fn=image_processing, inputs=inputs, outputs=outputs)
iface.launch()
