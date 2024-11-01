from pylibCZIrw import czi
from os import walk, path
import numpy as np
import cv2
import sys


if sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print("Usage: python main.py <directory>")
    exit()

# get all files in directory
filenames = next(walk(sys.argv[1]), (None, None, []))
dirname = filenames[0] if filenames[0] else sys.argv[1]
filenames = filenames[2]

total_area = 0
for filename in filenames:
    with czi.open_czi(path.join(dirname, filename)) as czi_file:
        # Read image
        frame = czi_file.read(pixel_type="Bgr48")

        # Prepare image for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)
        blurred = cv2.convertScaleAbs(blurred)

        # Calculate thresholds for Canny
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        edges = cv2.Canny(blurred, lower, upper)

        # Display the image
        cv2.imshow("Edges", edges)
        while True:
            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
