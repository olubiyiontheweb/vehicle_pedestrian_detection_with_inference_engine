import cv2

# specify path to model files
model_xml = "models/intel/pedestrian-and-vehicle-detector-adas-0001/FP16/pedestrian-and-vehicle-detector-adas-0001.xml"
model_bin = "models/intel/pedestrian-and-vehicle-detector-adas-0001/FP16/pedestrian-and-vehicle-detector-adas-0001.bin"

# read image or frame
img_way = cv2.imread("pedestrians.jpg")
print(img_way.shape)

# load the deep neural network optimized model files
model_net = cv2.dnn.readNet(model_xml, model_bin)

# Pre-process the image and resize the image to 384x672
blob = cv2.dnn.blobFromImage(img_way, size=(672, 384))
# Check the blob shape
print(blob.shape)

# Set the image as network model input
model_net.setInput(blob)
# Carry a forward propagation
out = model_net.forward()


# Check output shape
print("output shape: " + str(out.shape))

# Compare this with [1,1,N,7]
print("Number of objects found = {}".format(out.shape[2]))

# Reshape the output
detection = out.reshape(-1, 7)
print("Detection shape: " + str(detection.shape))
print("Detected labels: " + str(detection[:, 1]))

# print(detection)

for detectedObject in detection:
    # Find vehicle label
    label = int(detectedObject[1])
    # Choose color of bounding box, detected person
    if label == 2:
        # Green color
        color = (0, 255, 0)
    else:
        # Red color
        color = (0, 0, 255)
    # Find confidence
    confidence = float(detectedObject[2])
    # Bounding box coordinates
    xmin = int(detectedObject[3] * img_way.shape[1])
    ymin = int(detectedObject[4] * img_way.shape[0])

    xmax = int(detectedObject[5] * img_way.shape[1])
    ymax = int(detectedObject[6] * img_way.shape[0])
    # Plot bounding box only if there is at least 30% confidence
    if confidence >= 0.30:
        cv2.rectangle(img_way, (xmin, ymin), (xmax, ymax), color, 2)

# Display image
cv2.imshow("Output Image", img_way)
cv2.imwrite("pedestrian_image.png", img_way)
cv2.waitKey(0)
cv2.destroyAllWindows()