from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from ml_model import analyze_and_predict, predict_next_semester, clean_data
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        subject_means, suggestions, plot_paths = analyze_and_predict(data)
        predictions = predict_next_semester(data)
        return render_template('result.html', subject_means=subject_means, suggestions=suggestions, plot_paths=plot_paths, predictions=predictions)
    return redirect(request.url)

if __name__ == "__main__":
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    app.run(debug=True)
