import base64
from io import BytesIO
from flask import Flask, render_template, request
from PIL import Image
from utils import predict_image  # 🔁 Uses actual model now

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_data = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert("RGB")

            # Convert image to base64 to preserve preview
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode()

            prediction = predict_image(image)

    return render_template('index.html', prediction=prediction, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)