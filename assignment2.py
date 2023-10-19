import cv2
import numpy as np

image = cv2.imread('Image_01.bmp', cv2.IMREAD_GRAYSCALE)
neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

def labelObjects(image):
    def find_root(label):
        while parent[label] != label:
            label = parent[label]
        return label

    def union(label1, label2):
        root1 = find_root(label1)
        root2 = find_root(label2)
        if root1 != root2:
            parent[root1] = root2

    height, width = image.shape
    labels = np.zeros((height, width), dtype=int)
    parent = list(range(height * width))
    label = 0

    for x in range(height):
        for y in range(width):
            if image[x, y] == 255:
                neighborLabels = []
                for dx, dy in neighbors:
                    if 0 <= x + dx < height and 0 <= y + dy < width:
                        neighborLabel = labels[x + dx, y + dy]
                        if neighborLabel > 0:
                            neighborLabels.append(neighborLabel)

                if not neighborLabels:
                    label += 1
                    labels[x, y] = label
                else:
                    minLabel = min(neighborLabels)
                    labels[x, y] = minLabel
                    for neighborLabel in neighborLabels:
                        if neighborLabel != minLabel:
                            union(minLabel, neighborLabel)

    for x in range(height):
        for y in range(width):
            if image[x, y] == 255:
                labels[x, y] = find_root(labels[x, y])

    return labels, label

labeledImage, numObjects = labelObjects(image)
colors = np.random.randint(0, 255, size=(numObjects, 3), dtype=np.uint8)
outputImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

for label in range(1, numObjects + 1):
    outputImage[labeledImage == label] = colors[label - 1]

cv2.imshow('Labeled Objects', outputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
