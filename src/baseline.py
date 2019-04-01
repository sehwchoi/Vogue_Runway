# import OpenCV file
import cv2
import os
import numpy as np


input_folder = "./test_img"
input_files = []
for root, dirs, files in os.walk(input_folder):
    print(root, dirs, files)
    for f in files:
        if os.path.splitext(f)[-1].lower() in [".jpg", ".JPG"]:
            input_files.append([root, f])


cv_imgs = []
for file in input_files:
    # Read Image1
    full_path = os.path.join(file[0], file[1])
    print(full_path)
    cv_img = cv2.imread(full_path, 1)
    cv_imgs.append(cv_img)

weight = 1/len(input_files)
print("weight", weight)
# Blending the images with 0.3 and 0.7


#img_height, img_width = cv_imgs[0].shape[:2]
#n_channels = 3
#result_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
result_img = cv_imgs[0]

for img in cv_imgs[1:]:
    new_img = cv2.addWeighted(result_img, weight, img, weight, 0)
    result_img = new_img
    print("processed", img)


# Show the image
cv2.imshow('result_image', result_img)

# Wait for a key
cv2.waitKey(0)

# Distroy all the window open
cv2.distroyAllWindows()
