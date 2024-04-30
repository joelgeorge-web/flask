# Save this as app.py in your Flask app directory
import tensorflow as tf
import os
import cv2
import numpy as np
import random
import time
from flask import Flask, request, send_file,render_template,make_response
import io
from werkzeug.utils import secure_filename
from pygame import mixer
from flask_cors import CORS
import json, base64
# Initialize the Flask app
app = Flask(__name__)
CORS(app)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('best_model.h5')


emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
category = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

music_playing = False
start_time = None

# Route to render the camera page
@app.route('/')
def index():
    return render_template('index.html')  # Render the camera.html template

@app.route('/mood')
def mood():
    return render_template('mood.html')  # Render the camera.html template

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    image_file = request.files['file']
    image_data = image_file.read()

    frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        prediction = model.predict(roi)[0]
        emotion = emotions[prediction.argmax()]
        label = category[emotion]
        print(label)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Add text

        # Music handling if sad emotion detected
        # if label == 'sad':
        #     start_time = time.time()
        #     music_playing = True
        #     mixer.music.load(f'audio/{random.randrange(1, 13)}.mp3')
        #     mixer.music.play()

    # Encode and send back the annotated frame
    # _, buffer = cv2.imencode('.jpg', frame)
    # response = BytesIO(buffer.tobytes())
    # response.seek(0)
    
    ret, buffer = cv2.imencode('.jpg', frame)
    image_blob = io.BytesIO(buffer.tobytes())
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    # Set response headers for image content

    response_data = {
        "image_base64": image_base64,
        "isSad": True if label == 'sad' else False 
    }

    response = make_response(response_data, 200)
    response.headers.set('Content-Type', 'application/json')  # Adjust for actual format

    return response
    # return response
    # return send_file(response, mimetype='image/jpeg')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Use debug=True for development, set to False in production
