from pylibCZIrw import czi
from os import walk, path
import numpy as np
import cv2
import sys


def main():
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print("Usage: python main.py <directory>")
        exit()

    # get all files in directory
    filenames = next(walk(sys.argv[1]), (None, None, []))
    dirname = filenames[0] if filenames[0] else sys.argv[1]
    filenames = filenames[2]

    for filename in filenames:
        with czi.open_czi(path.join(dirname, filename)) as czi_file:

            # Read image
            # Possible values are: Gray8, Gray16, Gray32Float, Bgr24, Bgr48, Bgr96Float
            frame = czi_file.read(pixel_type="Bgr48").astype(np.uint8)

            # Prepare image for edge detection
            show(frame)
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(src=grey, ksize=(7, 7), sigmaX=0.5, borderType=cv2.BORDER_REFLECT)
            lap = cv2.Laplacian(blur, cv2.CV_8UC1)
            # sobel = cv2.Sobel(src=lap, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=5)
            show(lap)
            blur = cv2.GaussianBlur(lap, (5, 5), 0)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # thresh = cv2.adaptiveThreshold(lap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                cv2.THRESH_BINARY, 11, 7)

            edges = thresh
            # edges = cv2.Sobel(src=thresh, ddepth=cv2.CV_8UC1, dx=1, dy=1, ksize=5)
            show(edges)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bin_img = cv2.morphologyEx(thresh,
                                       cv2.MORPH_CLOSE,
                                       kernel,
                                       iterations=3)
            show(bin_img)

            # Find contours and draw them
            cntrs, _heir = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
            # approx = cv2.approxPolyDP(cntrs[0], 0.1 * cv2.arcLength(cntrs[0], True), True)
            edges = cv2.drawContours(bin_img, cntrs, -1, (0,255,0))
            show(edges)


def show(img):
    cv2.imshow("", img)
    while True:
        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def canny(img):
    v = np.median(img)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    return cv2.Canny(img, lower, upper)


if __name__ == "__main__":
    main()
