# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
fidan_image = face_recognition.load_image_file("00000.png")
fidan_face_encoding = face_recognition.face_encodings(fidan_image)[0]

# Load a second sample picture and learn how to recognize it.
bahar_image = face_recognition.load_image_file("img.JPG")
bahar_face_encoding = face_recognition.face_encodings(bahar_image)[0]

# Load a third sample picture and learn how to recognize it.
#alisha_image = face_recognition.load_image_file(" ")
#alisha_face_encoding = face_recognition.face_encodings(alisha_image)[0]

# Load a fourth sample picture and learn how to recognize it.
#nazila_image = face_recognition.load_image_file(" ")
#nazila_face_encoding = face_recognition.face_encodings(nazila_image)[0]

# Load a fifth sample picture and learn how to recognize it.
#elnara_image = face_recognition.load_image_file(" ")
#elnara_face_encoding = face_recognition.face_encodings(elnara_image)[0]

# Load a sixth sample picture and learn how to recognize it.
#nasib_image = face_recognition.load_image_file(" ")
#nasib_face_encoding = face_recognition.face_encodings(nasib_image)[0]

# Load a seventh sample picture and learn how to recognize it.
#ilyas_image = face_recognition.load_image_file(" ")
#ilyas_face_encoding = face_recognition.face_encodings(ilyas_image)[0]

# Load a eight sample picture and learn how to recognize it.
#matin_image = face_recognition.load_image_file(" ")
#matin_face_encoding = face_recognition.face_encodings(matin_image)[0]

# Load a nineth sample picture and learn how to recognize it.
#nuri_image = face_recognition.load_image_file(" ")
#nuri_face_encoding = face_recognition.face_encodings(nuri_image)[0]

# Load a tenth sample picture and learn how to recognize it.
zamig_image = face_recognition.load_image_file("00001.png")
zamig_face_encoding = face_recognition.face_encodings(zamig_image)[0]

# Load a second sample picture and learn how to recognize it.
#bahar_image = face_recognition.load_image_file("img.JPG")
#bahar_face_encoding = face_recognition.face_encodings(bahar_image)[0]

# Load a second sample picture and learn how to recognize it.
#bahar_image = face_recognition.load_image_file("img.JPG")
#bahar_face_encoding = face_recognition.face_encodings(bahar_image)[0]

# Load a second sample picture and learn how to recognize it.
#bahar_image = face_recognition.load_image_file("img.JPG")
#bahar_face_encoding = face_recognition.face_encodings(bahar_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    fidan_face_encoding,
    bahar_face_encoding,
    #alisha_face_encoding,
    #nazila_face_encoding,
    #elnara_face_encoding,
    #nasib_face_encoding,
    #ilyas_face_encoding,
    #matin_face_encoding,
    #nuri_face_encoding,
    zamig_face_encoding
]
known_face_names = [
    "Eldeniz",
    "Bahar Mahmudlu",
    #"Alisha Izabakarova",
    #"Nazila Habibzadeh",
    #"Elnara Mammadzada",
    #"Nasib Ahmedov",
    #"Ilyas Quluzada",
    #"Matin Qasimov",
    #"Nurlan Nuri",
    "Zamig Asgerzadeh"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
