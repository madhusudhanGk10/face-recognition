import cv2
import face_recognition
import numpy as np
import json

# Load the known faces and names from the JSON file
with open("known_faces.json", "r") as f:  # Replace with the path to your JSON file
    data = json.load(f)

# Initialize lists for face encodings and names
known_face_encodings = []
known_face_names = []

# Load images and their names from the JSON data
for person in data["known_faces"]:
    image_path = person["image_path"]
    name = person["name"]

    # Load the image of the known person
    person_image = face_recognition.load_image_file(image_path)
   
    # Extract face encoding
    person_face_encoding = face_recognition.face_encodings(person_image)[0]
   
    # Add the encoding and name to the lists
    known_face_encodings.append(person_face_encoding)
    known_face_names.append(name)

# Initialize the video capture (0 for webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop over each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default name is "Unknown"
       
        # If a match is found, assign the name of the recognized person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Put the name of the recognized person above the face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame with recognized faces
    cv2.imshow('Video', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()