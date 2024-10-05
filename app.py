from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import numpy as np
import wave
import io
import joblib
import preprocess_audio
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import jwt
import datetime
from functools import wraps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management
SECRET_KEY = 'your_secret_key'  # Change this to a secure secret key

# Load the trained model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Endpoint to register a new user.
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            conn.close()
            flash("User registered successfully!", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "error")

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Endpoint for user login.
    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT password FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            token = jwt.encode({'user': username, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, SECRET_KEY)
            return jsonify({"token": token}), 200
        else:
            flash("Invalid username or password", "error")

    return render_template('login.html')

def token_required(f):
    """
    Decorator to protect routes requiring a valid token.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 403

        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except:
            return jsonify({'message': 'Token is invalid!'}), 403

        return f(*args, **kwargs)

    return decorated

@app.route('/verify', methods=['POST'])
@token_required
def verify():
    """
    Endpoint to verify the speaker identity from an uploaded audio file.
    """
    file = request.files['audio']
    audio_bytes = file.read()
    
    # Save the uploaded file as a temporary .wav file
    with open('temp.wav', 'wb') as f:
        f.write(audio_bytes)
    
    # Preprocess the audio and extract features
    mfcc = preprocess_audio.preprocess_audio('temp.wav')
    features = np.array(mfcc).reshape(1, -1)
    features = scaler.transform(features)
    
    # Perform prediction
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    # Return the prediction and confidence as JSON
    return jsonify({'speaker': prediction[0], 'confidence': probabilities.max()})

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(debug=True)
