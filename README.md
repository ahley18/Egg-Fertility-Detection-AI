# Egg-Fertility-Detection-AI

This project demonstrates how to capture images from a webcam, process them using OpenCV and `skimage`, and apply various image processing techniques such as adaptive histogram equalization, Gaussian blur, thresholding, and contour detection. Additionally, it includes functionality to detect green color regions in the processed image, determining "fertility" status based on the presence of green.

## Features

- **Webcam Capture**: Captures frames from a connected webcam.
- **Save Captured Image**: Press the spacebar to capture and save an image from the video stream.
- **Image Processing**:
  - Adaptive histogram equalization for contrast enhancement.
  - Gaussian blur to smooth the image.
  - Thresholding for binary image creation.
  - Contour detection and drawing.
  - Cropping a specific region of interest for further analysis.
- **Green Color Detection**: Detects green color in the cropped region and returns a "Fertile" or "Infertile" status based on the presence of green.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Scikit-Image (`skimage`)
- Numpy
- Matplotlib

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your-username/webcam-image-processing.git
    cd webcam-image-processing
    ```

2. Install the required packages:

    ```bash
    pip install opencv-python scikit-image numpy matplotlib
    ```

## How to Run

1. Run the script:

    ```bash
    python your_script_name.py
    ```

2. The webcam feed will be displayed in a window.
    - Press the **spacebar** to capture and save the current frame.
    - Press **'q'** to quit the webcam stream.

3. After capturing an image, the program will:
    - Process the image (apply histogram equalization, blur, threshold, contour detection).
    - Detect green color in the cropped image region.
    - Display the results in both a window and a matplotlib plot.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
