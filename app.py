from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import threading

app = Flask(__name__)
camera = cv2.VideoCapture(0)


Ahaanaf_image = face_recognition.load_image_file("C:/Users/19789/Desktop/SNHU/data science/Flask/Flask FaceRecognition/JesusChrist/Ahaanaf.jpg")
Ahaanaf_face_encoding = face_recognition.face_encodings(Ahaanaf_image)[0]


known_face_encodings = [Ahaanaf_face_encoding]
known_face_names = ["Ahaanaf"]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frame():
    video_capture = cv2.VideoCapture(0)  # Access the webcam

    while True:
        # Capture a single frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Detect face encodings
        face_encodings = []
        try:
            if face_locations:
                # Generate encodings for detected face locations
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        except Exception as e:
            print(f"Error during face encoding: {e}")

        # Draw rectangles around detected faces
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations to match the original frame size
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    video_capture.release()


# Flask route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask app in a separate thread
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
