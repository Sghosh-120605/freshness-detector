from flask import Flask, render_template, request
import os
from utils.predict import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Only returning confidence + days_remaining now
        confidence, days_remaining, result_path = predict_image(image_path)
        return render_template(
            'result.html',
            confidence=f"{confidence:.2f}%",
            days_remaining=days_remaining,
            result_image=result_path
        )

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    app.run(debug=True)
