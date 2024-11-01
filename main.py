import cv2
from pylibCZIrw import czi
import sys
from os import walk, path


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
        frame = czi_file.read()
        print(frame)
