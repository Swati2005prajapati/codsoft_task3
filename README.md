# 👀 Face Detection, Recognition & Analysis Web App

## 📌 Overview
This project is part of my **CodSoft AI/ML Internship (August 2025)**.  
It is a Flask-based application that performs **real-time face detection, recognition, and analysis (age + expression prediction)**.

---

## ⚡ Features
- 👀 Face Detection (real-time using OpenCV)  
- 🧑‍🤝‍🧑 Face Recognition (identify known faces)  
- 📊 Face Analysis (predict **Age** & **Expression**)  
- 🎥 Live Camera Feed with bounding boxes  
- 📂 Store & manage face datasets  
- 🖥️ User-friendly web interface with Flask  

---

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript  
- **Backend**: Flask (Python)  
- **Libraries**: OpenCV, TensorFlow/Keras, NumPy, dlib  
- **Database/Storage**: Local file system for datasets  

---

## 📂 Project Structure
```

├── app.py                 # Flask main application
├── templates/             # HTML templates (home, detection, recognition, analysis)
│   ├── home.html
│   ├── detection.html
│   ├── recognization.html
│   ├── analysis.html
├── static/                # CSS & JS
│   ├── css/
│   │   └── style.css
│   ├── uploads/           # Uploaded images
├── data/                  # Face datasets
│   ├── known\_face/
│   ├── unknown\_face/
├── trainer/               # Model training data
│   ├── labels.json
│   ├── trainer.yml
├── assets/                # Screenshots
│   ├── homepage.png
│   ├── face\_detection.png
│   ├── face\_recognization.png
│   ├── face\_analysis.png
└── README.md

````

---

## 🚀 How to Run

1️⃣ Clone this repo:
```bash
git clone https://github.com/Swati2005prajapati/face_app.git
cd face_app
````

2️⃣ Create virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

4️⃣ Run the app:

```bash
python app.py
```

Server will start at:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📸 Screenshots

* [Homepage](https://raw.githubusercontent.com/Swati2005prajapati/face_app/main/assets/homepage.png)
* [Face Detection](https://raw.githubusercontent.com/Swati2005prajapati/face_app/main/assets/face_detection.png)
* [Face Recognition](https://raw.githubusercontent.com/Swati2005prajapati/face_app/main/assets/face_recognization.png)
* [Face Analysis](https://raw.githubusercontent.com/Swati2005prajapati/face_app/main/assets/face_analysis.png)

---

## 🔗 Links

* 📂 GitHub Repo: [face\_app](https://github.com/Swati2005prajapati/face_app)
* 🎥 LinkedIn Post: (link to task completion video)
* 🌐 CodSoft: [CodSoft Internship](https://codsoft.in)

---

## 🙏 Acknowledgements

Thanks to **CodSoft** for this internship opportunity.
This project improved my skills in **Flask, Computer Vision, and AI-based face analysis**.


Tamne hu aama ek **Play Online / Demo link (GitHub Pages style)** add karu ke face app ma flask chhe etle skip kariye?
```
