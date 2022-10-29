
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image_sat.jpg")

  

image = cv2.resize(img, (700, 600))
result_red = image.copy()
result_blue = image.copy()
result_green = image.copy()

# Convert Image to Image HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
rev_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

green = np.uint8([[[0, 255, 0]]])#basically we're just putting the rgb value here to get a reference of around what value hsv values will be there
#if you guys see that article we can take many rgb values for diff regions
  
# Convert Green color to Green HSV
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
  
# Print HSV Value for Green color
print(hsv_green)
#then look at that hsv image ill send and you can define the upper and lower range
  
# Defining lower and upper bound HSV values
lower_green = np.array([30, 100, 20])#60 is the actual hsv value
upper_green = np.array([80, 255, 255])

#v value from 20 to 255 and other values from the graph
#i read the above on some stack overflow page not too sure though

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# define range of blue color in HSV
lower_red = np.array([160,50,50])
upper_red = np.array([180,255,255]) 
# Defining mask for detecting color
mask_green = cv2.inRange(hsv, lower_green, upper_green)
rev_mask_green = cv2.cvtColor(mask_green, cv2.COLOR_BGR2RGB)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
rev_mask_blue = cv2.cvtColor(mask_blue, cv2.COLOR_BGR2RGB)
mask_red = cv2.inRange(hsv, lower_red, upper_red)
rev_mask_red = cv2.cvtColor(mask_red, cv2.COLOR_BGR2RGB)
result_red = cv2.bitwise_and(result_red, result_red, mask=mask_red)
rev_result_red = cv2.cvtColor(result_red, cv2.COLOR_BGR2RGB)
result_blue = cv2.bitwise_and(result_blue, result_blue, mask=mask_blue)
rev_result_blue = cv2.cvtColor(result_blue, cv2.COLOR_BGR2RGB)
result_green = cv2.bitwise_and(result_green, result_green, mask=mask_green)
rev_result_green = cv2.cvtColor(result_green, cv2.COLOR_BGR2RGB)
'''
# Display Image and Mask
cv2.imshow("Image", image)
cv2.imshow("Mask_Green", mask_green)
cv2.imshow("Mask_Blue", mask_blue)
cv2.imshow("Mask_Red", mask_red)
cv2.imshow('result_red', result_red)
cv2.imshow('result_blue', result_blue)
cv2.imshow('result_green', result_green)
'''
# create figure
fig = plt.figure(figsize=(10, 7))


# setting values to rows and column variables
rows = 3
columns = 3

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(rev_image)
plt.axis('off')
plt.title("Original Image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(rev_mask_green)
plt.axis('off')
plt.title("Mask Green")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(rev_result_green)
plt.axis('off')
plt.title("Result")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)


# showing image
plt.imshow(rev_image)
plt.axis('off')
plt.title("Original Image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(rev_mask_blue)
plt.axis('off')
plt.title("Mask Blue")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 6)

# showing image
plt.imshow(rev_result_blue)
plt.axis('off')
plt.title("Result")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 7)

# showing image
plt.imshow(rev_image)
plt.axis('off')
plt.title("Original Image")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 8)

# showing image
plt.imshow(rev_mask_red)
plt.axis('off')
plt.title("Mask Red")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 9)

# showing image
plt.imshow(rev_result_red)
plt.axis('off')
plt.title("Result")
plt.show()
cv2.waitKey(0) #wait until key is pressed
cv2.destroyAllWindows()

#basically we can just define hsv ranges for many colours detect the areas and display all the masked images in one file or something

#research paper
#https://www.researchgate.net/publication/342322385_Image_Processing_Techniques_for_Analysis_of_Satellite_Images_for_Historical_Maps_Classification-An_Overview
