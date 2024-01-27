# Changed from anaconda to virtualenv

import gradio as gr
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import median_filter
from skimage.filters import threshold_multiotsu, threshold_otsu

def basic_preprocessing(input_image, kernel_size):
    """
    Preprocessing: average, median filter, and histogram equalization with size and
    shape of the mask should be selected.((kernel_size = 3 by default))
    """
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
    '''
    Input: image and corresponding mask after normalization [0, 255] -> [0, 1], for example
    Output: metrics
    '''

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

def main_function(img, kernel_sz, original_mask):
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

    # metrics calculation
    sens, spec, acc = calculate_metrics(otsu_seg_mask, original_mask)
    metric_overall = f'Otsu: {sens:.2f}, {spec:.2f}, {acc:.2f}'

    return info, average_img, median_img, hist_equ_img, psnr, kapur_img, otsu_img, metric_overall

inputs = [
    gr.Image(type='numpy', label="Input original image"),
    gr.Number(label='kernel size (average and median filter only)', value=3),
    gr.Image(type='numpy', label="Input original mask"),
]
outputs = [
    gr.Textbox(label="Shape"),  # img info
    gr.Image(label="avg filtering"),  # avg filtering
    gr.Image(label="median filtering"),  # median filtering
    gr.Image(label="histogram equalized img"),  # histogram equalizing
    gr.Number(label="psnr value"),  # psnr value
    gr.Image(label="kapur_segmentation"),  # segmentation Kapur method
    gr.Image(label="otsu_segmentation"),  # segmentation Otsu method
    gr.Textbox(label="Metrics accuracy, specificity, sensitivity respectively"),  # metrics
]

iface = gr.Interface(fn=main_function, inputs=inputs, outputs=outputs)
iface.launch()
