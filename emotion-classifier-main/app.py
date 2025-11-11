

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
import base64
from PIL import Image
import io
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


EMOTION_EMOJIS = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Sad': '😢',
    'Surprise': '😲',
    'Neutral': '😐'
}


MODEL_PATH = 'emotion_model.h5'

print("Loading emotion detection model...")
try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✓ Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Could not load model: {e}")
    print("The app will use demo mode with random predictions")
    print("Please download the model using: python model.py")
    MODEL_LOADED = False


try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("✓ Face detector loaded successfully!")
except Exception as e:
    print(f"⚠️  Face detector error: {e}")


def init_db():
    conn = sqlite3.connect('emotion_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  emotion TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  source TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

def detect_and_predict_emotion(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, 0, {}
    
    
    (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
    
    
    face_roi = gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi.astype('float32') / 255.0
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = np.expand_dims(face_roi, axis=-1)
    
    
    if MODEL_LOADED:
        predictions = model.predict(face_roi, verbose=0)[0]
    else:
        
        predictions = np.random.dirichlet(np.ones(7))
    
    
    emotion_idx = np.argmax(predictions)
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = float(predictions[emotion_idx]) * 100
    
    all_predictions = {
        EMOTION_LABELS[i]: float(predictions[i] * 100) 
        for i in range(len(EMOTION_LABELS))
    }
    
    return emotion, confidence, all_predictions

def save_to_db(name, emotion, confidence, source):
    
    try:
        conn = sqlite3.connect('emotion_detection.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''INSERT INTO detections (name, timestamp, emotion, confidence, source)
                     VALUES (?, ?, ?, ?, ?)''',
                  (name, timestamp, emotion, confidence, source))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        
        image_data = data['image']
        name = data.get('name', 'Anonymous').strip() or 'Anonymous'
        source = data.get('source', 'upload')
        
      
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        
        emotion, confidence, all_predictions = detect_and_predict_emotion(image_cv)
        
        if emotion is None:
            return jsonify({
                'error': 'No face detected. Please ensure your face is clearly visible.'
            }), 400
        
        
        save_to_db(name, emotion, round(confidence, 2), source)
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'emoji': EMOTION_EMOJIS.get(emotion, '😐'),
            'all_predictions': all_predictions,
            'model_loaded': MODEL_LOADED
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    
    try:
        conn = sqlite3.connect('emotion_detection.db')
        c = conn.cursor()
        c.execute('''SELECT name, timestamp, emotion, confidence, source 
                     FROM detections ORDER BY id DESC LIMIT 50''')
        rows = c.fetchall()
        conn.close()
        
        data = [{
            'name': r[0],
            'timestamp': r[1],
            'emotion': r[2],
            'confidence': r[3],
            'source': r[4]
        } for r in rows]
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
   
    try:
        conn = sqlite3.connect('emotion_detection.db')
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM detections')
        total = c.fetchone()[0]
        
        c.execute('''SELECT emotion, COUNT(*) 
                     FROM detections GROUP BY emotion''')
        emotion_dist = dict(c.fetchall())
        
        c.execute('SELECT AVG(confidence) FROM detections')
        avg_conf = c.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total': total,
            'distribution': emotion_dist,
            'avg_confidence': round(avg_conf, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎭 EMOTION DETECTION WEB APP")
    print("="*60)
    print(f"Model: {'✓ Loaded' if MODEL_LOADED else '⚠️  Demo Mode'}")
    print("Server starting on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)