import cv2 as cv

split_path = "captcha-pict.png"

image = cv.imread(split_path)
(h, w) = image.shape[:2]

print('height : ', h)
print('widht : ', w)

threshold = (w//6)
errorness = 3

box1 = image[0:h, 0:threshold-7]
box2 = image[0:h, threshold:threshold*2-errorness]
box3 = image[0:h, threshold*2+5:threshold*3-errorness]
box4 = image[0:h, threshold*3+6:threshold*4-errorness]
box5 = image[0:h, threshold*4+9:threshold*5+5]
box6 = image[0:h, threshold*5+7:threshold*6+(w-threshold*6)]

cv.imshow('box1',box1)
cv.imshow('box2',box2)
cv.imshow('box3',box3)
cv.imshow('box4',box4)
cv.imshow('box5',box5)
cv.imshow('box6',box6)
cv.imwrite('code-captcha-split-1.jpg', box1)
cv.imwrite('code-captcha-split-2.jpg', box2)
cv.imwrite('code-captcha-split-3.jpg', box3)
cv.imwrite('code-captcha-split-4.jpg', box4)
cv.imwrite('code-captcha-split-5.jpg', box5)
cv.imwrite('code-captcha-split-6.jpg', box6)

cv.imshow('image', image)
cv.waitKey(0)