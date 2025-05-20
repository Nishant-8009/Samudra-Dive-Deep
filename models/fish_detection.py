

from ultralytics import YOLO

model = YOLO('/content/runs/detect/fish_small/weights/best.pt')

from PIL import Image
import matplotlib.pyplot as plt

# Provide the full path to the image
image_path = '/content/frame_11580_uinc.jpg'

# Open the image using PIL
image = Image.open(image_path)

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Single image path (ensure it's a string, not a list)
image_path = '/content/frame_11580_uinc.jpg'

# Run YOLO model prediction
yolo_outputs = model.predict(image_path)
output = yolo_outputs[0]
boxes = output.boxes
names = output.names

# Print detection results
print('**********************')
print(f'In this image, {len(boxes)} fish have been detected.')

for j in range(len(boxes)):
    label = names[boxes.cls[j].item()]
    coordinates = boxes.xyxy[j].tolist()
    confidence = np.round(boxes.conf[j].item(), 2)

    print(f'Fish {j + 1} is: {label}')
    print(f'Coordinates are: {coordinates}')
    print(f'Confidence is: {confidence}')
    print('-------')

# Get the annotated image (BGR to RGB)
annotated_image = output.plot()[:, :, ::-1]

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis('off')
plt.title("Detection Result")
plt.show()
