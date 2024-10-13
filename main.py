import cv2 as cv
from skimage import color, exposure
import numpy as np
import matplotlib.pyplot as plt


def take_picture_and_process_video(camera_index=0, file_name='img/img.png'):
    # Open the camera
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return

    while True:
        # Read frame by frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture frame from the camera.")
            break

        # Show the video stream
        cv.imshow('Webcam Video', frame)

        # Wait for key press
        key = cv.waitKey(1) & 0xFF

        # Press spacebar (' ') to capture and save the image
        if key == ord(' '):
            cv.imwrite(file_name, frame)
            print("Image captured and saved as", file_name)
            process_image(frame, file_name)  # Pass the frame and file name to the processing function

        # Press 'q' to quit the video stream
        elif key == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv.destroyAllWindows()


def process_image(frame, image_path):
    # Convert the captured image to RGB
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Apply adaptive histogram equalization
    image_equalized = exposure.equalize_adapthist(image_rgb, clip_limit=0.02)
    image_gray = color.rgb2gray(image_equalized)
    image_gray_8bit = (image_gray * 255).astype('uint8')

    # Gaussian Blur
    blurred = cv.GaussianBlur(image_gray_8bit, (5, 5), 0)

    # Apply thresholding
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Detect contours
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_img = frame.copy()
    cv.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

    # Save the contour image
    contour_img_path = 'img/contour_img.png'
    cv.imwrite(contour_img_path, contour_img)

    # Crop the image to a specified region
    contour_cropped_img = contour_img[200:350, 250:400]

    # Save the cropped image
    cropped_img_path = 'img/cropped_img.png'
    cv.imwrite(cropped_img_path, contour_cropped_img)

    # Check for green color in the cropped image
    fertile_status = "Fertile" if is_green_present(contour_cropped_img) else "Infertile"

    # Display the result
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True, sharey=True)

    # Display the processed image with contours
    ax1.imshow(cv.cvtColor(contour_img, cv.COLOR_BGR2RGB))
    ax1.set_title('Processed Image with Contours')

    # Add the fertility status text to the processed image
    ax1.text(0.5, 0.95, fertile_status, fontsize=35, ha='center', va='top',
             bbox=dict(facecolor='red', alpha=1, edgecolor='none'))

    # Display the original image
    ax2.imshow(image_rgb)
    ax2.set_title('Original Image')

    # Set axis labels off for both images
    ax1.axis('off')
    ax2.axis('off')

    image_data = cv.imread('img/cropped_img.png', cv.IMREAD_GRAYSCALE)  # Read the image as grayscale

    # Show the image in a window
    cv.imshow('Display Window', image_data)
    # Show the plot
    plt.show()


def is_green_present(cropped_img):
    # Convert the cropped image to HSV color space
    hsv_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])  # Adjust these values for better accuracy
    upper_green = np.array([80, 255, 255])

    # Create a mask to detect green regions
    green_mask = cv.inRange(hsv_img, lower_green, upper_green)

    # Check if any green pixels are present in the mask
    return cv.countNonZero(green_mask) > 0


# Call the function to start the webcam, capture an image, and process it
take_picture_and_process_video()
