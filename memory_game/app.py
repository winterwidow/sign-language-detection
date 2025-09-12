from flask import Flask, render_template, request, jsonify, session
import random
import string
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

LETTERS = list(string.ascii_uppercase)

DIFFICULTIES = {
    "easy": {"base_length": 3, "display_time": 1500},
    "medium": {"base_length": 4, "display_time": 1200},
    "hard": {"base_length": 5, "display_time": 1000}
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_game", methods=["POST"])
def start_game():
    """
    Starts a new game at the chosen difficulty.
    Initializes session state for the game.
    """
    try:
        data = request.get_json()
        difficulty = data.get("difficulty", "easy")
        
        # Initialize session state
        session['difficulty'] = difficulty
        session['score'] = 0
        session['level'] = 1
        session['sequence'] = generate_sequence(difficulty, round_number=1)
        session['current_index'] = 0
        session['game_active'] = True
        
        return jsonify({
            "sequence": session['sequence'],
            "score": session['score'],
            "level": session['level'],
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/next_round", methods=["POST"])
def next_round():
    """
    Proceeds to the next round with increased difficulty.
    """
    try:
        data = request.get_json()
        round_number = data.get("round", session.get('level', 1))
        difficulty = session.get('difficulty', 'easy')
        
        # Update session state
        session['level'] = round_number
        session['sequence'] = generate_sequence(difficulty, round_number)
        session['current_index'] = 0
        session['game_active'] = True
        
        return jsonify({
            "sequence": session['sequence'],
            "score": session['score'],
            "level": session['level'],
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/predict_sign", methods=["POST"])
def predict_sign():
    """
    Mock AI sign language prediction.
    In a real implementation, this would process the camera frame
    and return the detected sign using a trained ML model.
    """
    try:
        data = request.get_json()
        frame = data.get("frame", "")
        expected = data.get("expected", "")
        
        if not session.get('game_active', False):
            return jsonify({
                "status": "game_not_active",
                "message": "Game is not currently active"
            })
        
        current_index = session.get('current_index', 0)
        sequence = session.get('sequence', [])
        
        if current_index >= len(sequence):
            return jsonify({
                "status": "sequence_complete",
                "message": "Sequence already completed"
            })
        
        expected_letter = sequence[current_index]
        
        # Mock AI prediction - randomly succeed/fail for demo
        # In reality, this would process the base64 image frame
        detected_letter = mock_sign_detection(frame, expected_letter)
        
        if detected_letter == expected_letter:
            session['current_index'] = current_index + 1
            session['score'] = session.get('score', 0) + 1
            
            # Check if sequence is complete
            if session['current_index'] >= len(sequence):
                session['game_active'] = False
                return jsonify({
                    "status": "sequence_complete",
                    "detected": detected_letter,
                    "expected": expected_letter,
                    "score": session['score']
                })
            else:
                return jsonify({
                    "status": "correct",
                    "detected": detected_letter,
                    "expected": expected_letter,
                    "score": session['score'],
                    "progress": f"{session['current_index']}/{len(sequence)}"
                })
        else:
            # Wrong sign detected - game continues but no progress
            return jsonify({
                "status": "wrong",
                "detected": detected_letter,
                "expected": expected_letter,
                "score": session['score'],
                "message": "Try again with the correct sign"
            })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/reset_game", methods=["POST"])
def reset_game():
    """
    Resets the current game session.
    """
    try:
        # Clear game session
        session.pop('difficulty', None)
        session.pop('score', None)
        session.pop('level', None)
        session.pop('sequence', None)
        session.pop('current_index', None)
        session.pop('game_active', None)
        
        return jsonify({
            "status": "success",
            "message": "Game reset successfully"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/game_status", methods=["GET"])
def game_status():
    """
    Returns current game status.
    """
    try:
        return jsonify({
            "game_active": session.get('game_active', False),
            "score": session.get('score', 0),
            "level": session.get('level', 1),
            "difficulty": session.get('difficulty', 'easy'),
            "current_index": session.get('current_index', 0),
            "sequence_length": len(session.get('sequence', [])),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

# ---------------- Helper Functions ----------------

def generate_sequence(difficulty, round_number=1):
    """
    Generates a random sequence of letters based on difficulty and round.
    """
    base_length = DIFFICULTIES.get(difficulty, DIFFICULTIES["easy"])["base_length"]
    # Progressive difficulty: add 1 letter every 2 rounds
    length = base_length + ((round_number - 1) // 2)
    return [random.choice(LETTERS) for _ in range(length)]

def mock_sign_detection(frame_data, expected_letter):
    """
    Mock AI sign detection function.
    Replace this with actual AI model prediction in production.
    
    For demo purposes:
    - 70% chance of detecting the correct sign
    - 30% chance of random incorrect detection
    """
    if not frame_data:
        return random.choice(LETTERS)
    
    # Simulate AI processing delay
    import time
    time.sleep(0.1)
    
    # Mock success rate based on "image quality" (frame data length)
    success_rate = 0.7 if len(frame_data) > 1000 else 0.5
    
    if random.random() < success_rate:
        return expected_letter
    else:
        # Return a different random letter
        wrong_letters = [l for l in LETTERS if l != expected_letter]
        return random.choice(wrong_letters)

def process_camera_frame(frame_data):
    """
    Process base64 camera frame for AI model input.
    This is where you'd integrate your actual sign language detection model.
    """
    try:
        # Decode base64 image
        header, encoded = frame_data.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array for model processing
        image_array = np.array(image)
        
        # Here you would:
        # 1. Preprocess the image (resize, normalize, etc.)
        # 2. Feed it to your trained sign language model
        # 3. Return the predicted letter
        
        return image_array
    
    except Exception as e:
        print(f"Error processing camera frame: {e}")
        return None

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "status": "error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)