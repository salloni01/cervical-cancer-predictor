from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import sqlite3
from datetime import datetime
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = 'static/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
img_model = load_model("image_model.keras")

def init_db():

    conn = sqlite3.connect("database.db")

    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT,

        email TEXT UNIQUE,

        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        email TEXT,

        risk_level TEXT,

        probability REAL,

        created_at TEXT
    )
    """)

    conn.commit()

    conn.close()


init_db()

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':

        username = request.form.get('username')

        email = request.form.get('email')

        password = request.form.get('password')

        conn = sqlite3.connect("database.db")

        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        existing_user = cur.fetchone()

        if existing_user:

            conn.close()

            return "Email already registered!"

        cur.execute("""
        INSERT INTO users (

            username,
            email,
            password

        )

        VALUES (?, ?, ?)
        """, (

            username,
            email,
            password
        ))

        conn.commit()

        conn.close()

        session['email'] = email

        return redirect(url_for('choose'))

    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():

    if request.method == 'POST':

        email = request.form.get('email')
        password = request.form.get('password')

        conn = sqlite3.connect("database.db")
        cur = conn.cursor()

        cur.execute("""
        SELECT * FROM users
        WHERE email=? AND password=?
        """, (email, password))

        user = cur.fetchone()
        conn.close()

        if user:
            session['email'] = email
            session['login_source'] = 'signin'
            return redirect(url_for('choose'))

        else:
            
            return render_template(
                'signin.html',
                error="Invalid Email or Password!"
            )

    return render_template('signin.html')

@app.route('/choose')
def choose():

    if 'email' not in session:

        return redirect(url_for('signin'))

    return render_template('choose.html')


@app.route('/details')
def details():

    if 'email' not in session:

        return redirect(url_for('signin'))

    return render_template('details.html')

@app.route('/image_upload')
def image_upload():

    if 'email' not in session:

        return redirect(url_for('signin'))

    return render_template('image_upload.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'email' not in session:
        return redirect(url_for('signin'))

    image = request.files.get('scan_image')

    if image.filename == "":
        return "No image selected"

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    img = keras_image.load_img(image_path, target_size=(224, 224))
    

    img_array = keras_image.img_to_array(img)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)

    
    prediction = img_model.predict(img_array)
    
    
    probability = int(float(prediction[0][0]) * 100)

   
    if probability > 70:
        risk_level = "High Risk"
        advice = [
            "Immediate medical consultation advised",
            "Advanced screening recommended",
            "Do not delay diagnosis"
        ]
    elif probability > 40:
        risk_level = "Medium Risk"
        advice = [
            "Regular monitoring advised",
            "Consult specialist soon",
            "Maintain healthy lifestyle"
        ]
    else:
        risk_level = "Low Risk"
        advice = [
            "Routine screening sufficient",
            "Maintain healthy habits",
            "Continue preventive care"
        ]

    session['risk_level'] = risk_level
    session['prob_percent'] = probability

    return render_template(
        'image_result.html',
        image_file=image.filename,
        probability=probability,
        risk_level=risk_level,
        advice=advice
    )
@app.route('/predict', methods=['POST'])
def predict():

    values = [
        float(request.form.get("Age", 0)),
        float(request.form.get("Number_of_sexual_partners", 0)),
        float(request.form.get("First_sexual_intercourse", 0)),
        float(request.form.get("Smokes", 0)),
        float(request.form.get("Smokes_years", 0)),
        float(request.form.get("Hormonal_Contraceptives", 0)),
        float(request.form.get("IUD", 0)),
        float(request.form.get("STDs", 0)),
        float(request.form.get("STDs_number", 0)),
        float(request.form.get("Dx_Cancer", 0)),
        float(request.form.get("Dx_CIN", 0)),
        float(request.form.get("Dx_HPV", 0))
    ]

    cols = [
        "Age", "Number of sexual partners", "First sexual intercourse",
        "Smokes", "Smokes (years)", "Hormonal Contraceptives", "IUD",
        "STDs", "STDs (number)", "Dx:Cancer", "Dx:CIN", "Dx:HPV"
    ]

    data = pd.DataFrame([values], columns=cols)
    data = imputer.transform(data)
    prob = model.predict_proba(data)[0][1]
    prob_percent = int(prob * 100)

    if prob > 0.7:
        risk_level = "High Risk"
        advice = [
            {"text": "Immediate consultation required"},
            {"text": "Advanced screening needed"},
            {"text": "Do not delay medical care"}
        ]
    elif prob > 0.4:
        risk_level = "Medium Risk"
        advice = [
            {"text": "Schedule doctor visit"},
            {"text": "Regular checkups needed"},
            {"text": "Avoid risk factors"}
        ]
    else:
        risk_level = "Low Risk"
        advice = [
            {"text": "Maintain healthy lifestyle"},
            {"text": "Routine screening"},
            {"text": "Stay active"}
        ]

    factors = [
        {"name": "Age", "contribution": max(0, min(100, int(values[0])))},
        {"name": "Sexual Partners", "contribution": max(0, min(100, int(values[1] * 10)))},
        {"name": "Smoking", "contribution": max(0, min(100, int(values[3] * 100)))},
        {"name": "STDs", "contribution": max(0, min(100, int(values[7] * 100)))},
        {"name": "Cancer History", "contribution": max(0, min(100, int(values[9] * 100)))}
    ]

    session['risk_level'] = risk_level
    session['prob_percent'] = prob_percent
    session['last_result'] = {
        "risk_level": risk_level,
        "prob_percent": prob_percent,
        "advice": advice,
        "factors": factors
    }

    return redirect(url_for('result'))
@app.route('/result')
def result():

    if 'last_result' not in session:

        return redirect(url_for('details'))

    data = session['last_result']

    return render_template(

        "result.html",

        risk_level=data["risk_level"],

        prob_percent=data["prob_percent"],

        advice=data["advice"],

        factors=data["factors"]
    )

@app.route('/save_result', methods=['GET','POST'])
def save_result():

    if 'email' not in session:

        return redirect(url_for('signin'))

    conn = sqlite3.connect("database.db")

    cur = conn.cursor()

    cur.execute("""
    INSERT INTO results (

        email,
        risk_level,
        probability,
        created_at

    )

    VALUES (?, ?, ?, ?)
    """, (

        session['email'],

        session['risk_level'],

        session['prob_percent'],

        datetime.now().strftime("%d-%m-%Y %H:%M")
    ))

    conn.commit()

    conn.close()

    return redirect(request.referrer)

@app.route('/history')
def history():

    if 'email' not in session:

        return redirect(url_for('signin'))

    conn = sqlite3.connect("database.db")

    cur = conn.cursor()

    cur.execute("""
    SELECT risk_level,
           probability,
           created_at

    FROM results

    WHERE email=?

    ORDER BY id DESC
    """, (

        session['email'],
    ))

    rows = cur.fetchall()

    conn.close()

    return render_template(

        "history.html",

        rows=rows
    )

@app.route('/compare')
def compare():

    if 'email' not in session:

        return redirect(url_for('signin'))

    conn = sqlite3.connect("database.db")

    cur = conn.cursor()

    cur.execute("""
    SELECT risk_level,
           probability,
           created_at

    FROM results

    WHERE email=?

    ORDER BY id DESC

    LIMIT 2
    """, (

        session['email'],
    ))

    rows = cur.fetchall()

    conn.close()

    if len(rows) < 2:

        return render_template(

            "compare.html",

            error="Not enough data to compare"
        )

    latest = rows[0]

    previous = rows[1]

    comparison = {

        "risk_change":
        f"{previous[0]} ->  {latest[0]}",

        "prob_change":
        float(latest[1]) - float(previous[1])
    }

    return render_template(

        "compare.html",

        latest=latest,

        previous=previous,

        comparison=comparison
    )
@app.route('/logout')
def logout():

    session.clear()

    return redirect(url_for('home'))

if __name__ == "__main__":

    app.run(debug=True)
