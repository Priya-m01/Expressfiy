from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import numpy as np
from PIL import Image
from io import BytesIO  # For handling byte streams
import base64  # For decoding the Base64 image data
import pandas as pd
from flask_mail import Mail, Message
from deepface import DeepFace
import cv2
from werkzeug.utils import secure_filename
from spotipy.oauth2 import SpotifyClientCredentials
import parselmouth
from parselmouth.praat import call
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import re
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.text import MIMEText

# Initialize Flask app, DB, and encryption
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
# db.init_app(app)

# set TF_ENABLE_ONEDNN_OPTS=0;

bcrypt = Bcrypt(app)
# db = SQLAlchemy(app)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    emotion=db.Column(db.String(100))
    feedback_date = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp of feedback

# Feedback model
class Feedback(db.Model):
    feed_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rating = db.Column(db.Integer)  # Storing rating (1 to 5 stars)
    comment = db.Column(db.String(500))  # Storing the comment
    email = db.Column(db.String(120), nullable=False)  # User's email (foreign key reference)
    feedback_date = db.Column(db.DateTime, default=datetime.utcnow)  

songs_df = pd.read_csv(r'C:\main\beatbuddy\facial2\data\Spotify_Song_Attributes.csv')


def recommend_songs(emotion):
    genre_map = {
        'happy': 'pop',
        'sad': 'pop', #'acoustic'
        'angry': 'rock',
        'surprise': 'dance',
        'neutral': 'classical',
        'fear': 'ambient',
        'disgust': 'metal',
        'calm': 'jazz',
        'excited': 'electronic'
    }
    genre = genre_map.get(emotion, 'pop')
    songs_df.rename(columns={'Genre': 'genre'}, inplace=True)  # Rename if necessary
    recommendations = songs_df[songs_df['genre'].str.contains(genre, na=False)].sample(10)
    return recommendations.apply(
        lambda row: {
            'trackName': row['trackName'],
            'artistName': row['artistName'],
            'url': f"https://open.spotify.com/track/{row['id']}" if 'id' in row else '#'
        },
        axis=1
    ).tolist()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        if User.query.filter_by(email=email).first():
            flash("Email already exists! Please login.", "danger")
            return redirect(url_for('login'))
        
        user = User(username=username, email=email, password=password )
        db.session.add(user)
        db.session.commit()

        flash("Signup successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash("Login successful!", "success")
            return redirect(url_for('welcome'))
        else:
            flash("Invalid email or password!", "danger")

        flash("Invalid credentials, Please try again.", "danger")
    return render_template('login.html')


@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    recommendations = songs_df.sample(n=10)[['trackName', 'artistName', 'url']].to_dict(orient='records')

    return render_template('welcome.html', username=session['username'], recommendations=recommendations)

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.get_json()
    image_data = data['image']

    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        emotion_result = DeepFace.analyze(img, actions=['emotion'])
        detected_emotion = emotion_result[0]['dominant_emotion']
    except Exception as e:
       return jsonify({'error': str(e)})

    recommendations = recommend_songs(detect_emotion)
    return jsonify({'emotion': detected_emotion, 'recommendations': recommendations})

#Speech input
@app.route('/analyze_pitch', methods=['POST'])
def analyze_pitch():
    try:
        data = request.get_json()
        speech_text = data['speechText']
        
        pitch_avg = 20 
        if pitch_avg > 250:
            detected_emotion = 'excited'
        elif pitch_avg < 150:
            detected_emotion = 'calm'
        else:
            detected_emotion = 'neutral'

        return jsonify({'emotion': detected_emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

# Text input 
@app.route('/analyze_text_tone', methods=['POST'])
def analyze_text_tone():
    try:
       
        data = request.get_json()
        text_input = data['text']

        
        if not re.match("^[a-zA-Z\u0900-\u097F\u0C80-\u0CFF\s]+$", text_input):
            return jsonify({'error': 'Input contains invalid characters. Only letters in English, Hindi, or Kannada are allowed.'}), 400

      
        translator = Translator()
        translated_text = translator.translate(text_input, src='auto', dest='en').text

        
        blob = TextBlob(translated_text)
        polarity = blob.sentiment.polarity

       
        if polarity > 0.5:
            detected_emotion = 'happy'
        elif polarity < -0.5:
            detected_emotion = 'sad'
        else:
            detected_emotion = 'neutral'

        return jsonify({'emotion': detected_emotion})
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing your request.', 'details': str(e)}), 500


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'POST':
       
        rating = request.form.get('rating')
        comment = request.form.get('comment')
        email = request.form.get('email')

        # user = User.query.filter_by(email=email).first()
        # new_feedback = Feedback(
        #     rating=rating,
        #     comment=comment,
        #     email=email,
        # )
        
    #     db.session.add(new_feedback)
    #     db.session.commit()
    #     # Compose email content
    #     subject = "Thank You for Your Feedback!"
    #     body = (
    #      f"Dear User,\n\n"
    #      f"Thank you for providing your feedback on our emotion-based song recommendation system!\n\n"
    #      f"Your Rating: {rating}\n"
    #      f"Your Comment: {comment}\n\n"
    #      f"We appreciate your valuable input and hope you continue enjoying our service.\n\n"
    #      f"Best regards,\n"
    #      f"The Expressify Team"
    #     )
    #    # Send email
    #     send_email(subject, body, email)
    #     flash("Thank you for your feedback! A confirmation email has been sent.", "success")
    # # return redirect(url_for('feedback_page'))  # Adjust to your feedback page route

        return redirect(url_for('home'))
  
    # Render the logout form
    return render_template('logout.html')

camera = cv2.VideoCapture(0)
# Release the camera when the app shuts down

@app.teardown_appcontext
def release_camera(exception=None):
   
    if camera.isOpened():
        camera.release()

@app.route('/recommend', methods=['GET'])
def recommend():
    emotion = request.args.get('emotion')
    recommendations =recommend_songs(emotion)
    return render_template('recommend.html', emotion=emotion, songs=recommendations)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

