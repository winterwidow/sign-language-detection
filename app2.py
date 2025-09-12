# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

from flask import Flask, render_template, Response, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import os

# Suppress specific warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__, static_folder='.', template_folder='.')
app.config['SECRET_KEY'] = 'secret!'

# Enable CORS for all routes
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)
    model = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/start')
def start():
    return send_from_directory('.', 'start.html')

@app.route('/start.html')
def start_html():
    return send_from_directory('.', 'start.html')

@app.route('/index.html')
def index_html():
    return send_from_directory('.', 'index.html')

@app.route('/gameSelector.html')
def game_selector():
    return send_from_directory('.', 'gameSelector.html')

# Health check endpoint
@app.route('/health')
def health_check():
    return {'status': 'online', 'message': 'GestureFlow AI server is running'}, 200

# Wake up endpoint (for auto-start attempts)
@app.route('/wake-up')
def wake_up():
    return {'status': 'awake', 'message': 'Server is ready'}, 200

# Start server endpoint
@app.route('/start-server', methods=['POST'])
def start_server_endpoint():
    return {'status': 'started', 'message': 'Server is already running'}, 200

# Serve static files (CSS, JS, images)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'msg': 'Connected to GestureFlow AI server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Check if camera is accessible
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

    labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
    }

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract hand landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Bounding box coordinates
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Make prediction if model is loaded
                if model is not None:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        confidence = max(prediction_proba[0])
                        predicted_character = labels_dict[int(prediction[0])]
                        
                        # Only emit if confidence is above threshold
                        if confidence > 0.7:  # 70% confidence threshold
                            socketio.emit('prediction', {
                                'text': predicted_character, 
                                'confidence': confidence
                            })
                        
                        # Draw bounding box and prediction
                        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # Green if confident, orange if not
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(frame, f"{predicted_character} ({confidence*100:.1f}%)", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(frame, "Error", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting GestureFlow server...")
    print("Access the application at: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)