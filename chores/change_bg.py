import cv2
img = cv2.imread('/Users/xbkaishui/Desktop/restart.png')

height, width, _ = img.shape
 
for i in range(height):
    for j in range(width):
        # img[i,j] is the RGB pixel at position (i, j)
        # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
        # print(f"i: {i}, j: {j}, img[i,j]: {img[i,j]}")
        if img[i,j].sum() < 50:
            img[i, j] = [255, 255, 255]

cv2.imwrite('/Users/xbkaishui/Desktop/restart_1.png', img)