from flask import Flask, render_template, request
import os
from utils.predict import predict_fruit_freshness  # ✅ updated import

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

        # ✅ Run adaptive prediction
        result = predict_fruit_freshness(image_path)

        return render_template(
            'result.html',
            fruit=result["fruit"],
            confidence=result["confidence"],
            use_within=result["use_within"],
            result_image=image_path
        )


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    app.run(debug=True)
