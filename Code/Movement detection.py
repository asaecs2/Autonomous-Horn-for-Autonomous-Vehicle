"""import cv2
import imutils

# Set up the capture object
cap = cv2.VideoCapture(0)

# Set up the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Set up variables to track the previous position of people
prev_rects = []

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Resize the frame
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    
    # Detect people in the frame
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    # Flag to indicate if motion was detected
    motion_detected = False
    
    # Draw bounding boxes around the people and check for motion
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for (x2, y2, w2, h2) in prev_rects:
            if abs(x - x2) > 10 or abs(y - y2) > 10:
                motion_detected = True
    
    # Print a message if motion was detected
    if motion_detected:
        print("Motion detected!")
    
    # Update the previous position of people
    prev_rects = rects
    
    # Show the result
    cv2.imshow('Frame', frame)
    
    # Check if the user pressed 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
"""

import cv2
import numpy as np

# Load the pre-trained HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Set up the video capture
video_capture = cv2.VideoCapture(0)  # Change the parameter to the appropriate video source if needed

while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()
    
    # Check if the frame was properly read
    if not ret:
        break
    
    # Resize the frame to a smaller size for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect people using HOG
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
    
    # Draw bounding boxes around the detected people
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
