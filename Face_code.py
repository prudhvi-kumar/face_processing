# live video stream

import cv2
cap = cv2.VideoCapture(0)
ret , photo = cap.read()

while True:
    ret , photo = cap.read()
    cv2.imshow("live video" , photo)
    if cv2.waitKey(10) == 13:
        break

cv2.destroyAllWindows()

---------------------------------------------------------------------------------------------------------------------------------------------------------------

# detecting human face from live video stream using rectangle

while True:
    ret , photo = cap.read()
    model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    mphoto = model.detectMultiScale(photo)
    if len(mphoto) == 0: # it helps when your face doesnt detect proprerly then there will be no coordinates then here it prints no faces detected which helps us to overcome the problem of out of index...
        print("no face detected")
    else: # if the face detected properly then the following code runs to detect face part
        x1 = mphoto[0][0]
        y1 = mphoto[0][1]
        x2 = mphoto[0][2] + x1
        y2 = mphoto[0][3] + y1
        rphoto = cv2.rectangle(photo , (x1,y1) , (x2,y2) , [255,0,0] , 5 )
    
        cv2.imshow("hey" , rphoto)
        if cv2.waitKey(10) == 13: #here 13 represents "ENTER" key
            break
cv2.destroyAllWindows()

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Identifying and getting face positions

model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, photo = cap.read()
    
    # Detect faces in the frame
    mphoto = model.detectMultiScale(photo)
    
    if len(mphoto) == 0:
        print("No face detected")
    else:
        # Get the coordinates of the detected face
        x1, y1, w, h = mphoto[0]
        x2, y2 = x1 + w, y1 + h
        
        # Draw rectangle around the detected face
        rphoto = cv2.rectangle(photo, (x1, y1), (x2, y2), [255, 0, 0], 5)
        
        # Add coordinates text to the image
        cv2.putText(rphoto, f"({x1}, {y1})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 255, 0], 2)
        cv2.putText(rphoto, f"({x2}, {y2})", (x2, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 255, 0], 2)

        cv2.imshow("Live Video Stream", rphoto)
    
    if cv2.waitKey(1) == 13:  # Press 'ENTER' to exit the loop
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

------------------------------------------------------------------------------------------------------------------------------------------------------------

# blurring the detected face and keeping as intact

import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Load the pre-trained face detection model
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, photo = cap.read()
    
    # Detect faces in the frame
    mphoto = model.detectMultiScale(photo)
    
    if len(mphoto) == 0:
        print("No face detected")
    else:
        for (x1, y1, w, h) in mphoto:
            # Create regions of interest (ROIs) for the face and the surrounding area
            face_roi = photo[y1:y1+h, x1:x1+w]
            surroundings_roi = photo[:y1, :]  # Region above the face
            
            # Apply Gaussian blur to the face region
            blurred_face = cv2.GaussianBlur(face_roi, (25, 25), 0)
            
            # Combine the ROIs to get the final frame with blurred face and surroundings intact
            photo[y1:y1+h, x1:x1+w] = blurred_face
            photo[:y1, :] = surroundings_roi
        
        # Add coordinates text to the image
        for (x1, y1, w, h) in mphoto:
            cv2.putText(photo, f"({x1}, {y1})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 255, 0], 2)
            cv2.putText(photo, f"({x1+w}, {y1+h})", (x1, y1 + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 255, 0], 2)

        cv2.imshow("Live Video Stream", photo)
    
    if cv2.waitKey(1) == 13:  # Press 'ENTER' to exit the loop
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
