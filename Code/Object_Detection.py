"""import cv2
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt') # load an official model

# Open a video capture object for the webcam
cap = cv2.VideoCapture(0)

# Check if the video capture object was opened successfully
if not cap.isOpened():
    print('Error: Failed to open video capture object')
    exit()

# Loop over the video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # Check if we reached the end of the video
    if not ret:
        break

    # Predict with the model
    results = model(frame) # predict on the current frame

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
"""


import cv2
from ultralytics import YOLO
import numpy as np
from openpyxl import Workbook as wb
import pandas as pd

# Load a model
model = YOLO('yolov8n.pt') # load an official model

# Open a video capture object for the webcam
cap = cv2.VideoCapture(0)

# Check if the video capture object was opened successfully
if not cap.isOpened():
    print('Error: Failed to open video capture object')
    exit()

# Open a file to store the object data
file = open("test1.txt", 'w')
i=0
# Create an empty DataFrame
df = pd.DataFrame()

# Save the DataFrame to an Excel file at a specific path
df.to_excel('"D:\test_1.xlsx"', index=False, engine='openpyxl')
# Loop over the video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # Check if we reached the end of the video
    if not ret:
        break

    # Predict with the model
    results = model(frame) # predict on the current frame
    #success = model.export(format='onnx')
    #print (success)
    # Assuming `results` is the variable that contains the output you provided
    #print(,result.0, "\n")
    # Write the object data to the file
    file.write(f"{results}")
    boxes = results[0].boxes
    #box = boxes[0]  # returns one box
    num=boxes.cls
    list_ = num.tolist()
    print(list_)
    #file.write(f"{list_}")
    #workbook = wb()
    #sheet = workbook.active
    # Write the numeric value to a cell
    #last_row = sheet.max_row + 1
    data_string = ', '.join(str(value) for value in list_)
    #cell = sheet.cell(row=last_row + i, column=1)  # Write in column A, change if needed
    #cell.value = data_string
    df["test1"][i]=data_string
    # Save the workbook
    #workbook.save("numeric_lists.xlsx")
    # Convert the list to a string
    #data_string = ', '.join(str(value) for value in list_)

    # Insert the string into a single cell
    #sheet['A1'].value = data_string

    # Display the frame
    cv2.imshow('frame', frame)
    i+=1
    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
workbook.save("output.xlsx")
cap.release()

# Close the file
file.close()
# Close all OpenCV windows
cv2.destroyAllWindows()


