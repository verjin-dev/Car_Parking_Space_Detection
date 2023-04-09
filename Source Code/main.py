import cv2
import numpy as np

# Load the YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Set the labels of the classes we want to detect (in this case, only one class: parking space)
classes = ["parking space"]

# Load the image and convert it to a blob
img = cv2.imread("parking_lot.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the YOLOv3 model and forward pass
net.setInput(blob)
output_layers = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_layers)

# Loop over the detected objects and draw bounding boxes around parking spaces
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] == "parking space":
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])
            h = int(detection[3] * img.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Parking Lot Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
