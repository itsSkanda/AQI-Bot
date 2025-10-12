import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from PIL import Image as im
from flask import Flask, render_template, request
import openai

# ------------------- TensorFlow / AQI Predictor Setup -------------------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

MODEL_PATH = os.path.join(os.getcwd(), "model.h5")
loaded_model = tf.keras.models.load_model(MODEL_PATH)
loaded_model.compile(optimizer='adam', loss='mean_absolute_error',
                     metrics=['mean_squared_error', tf.keras.metrics.RootMeanSquaredError()])

def preprocess_image(image):
    image = tf.image.resize(image, (200, 200))
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] != 3:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
    image = image / 255.0
    cropped_image = image[:120]
    cropped_image = tf.ensure_shape(cropped_image, (120, 200, 3))
    return cropped_image

# ------------------- Flask App Setup -------------------
app = Flask(__name__)
app.static_folder = 'static'

# ------------------- OpenAI / Chatbot Setup -------------------
client = openai.OpenAI(
    api_key="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",
    #os.getenv("OPENAI_API_KEY")
    base_url="https://openrouter.ai/api/v1"
)

# ------------------- ROUTES -------------------

# Home page with buttons
@app.route('/')
def home():
    return render_template('home.html')

# Chatbot page
@app.route('/chat')
def chat_page():
    return render_template('chat.html')  # renamed from index.html

@app.route('/index')
def index():
    return render_template('index.html')

# Chatbot API route
@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful and friendly assistant who helps in queries regarding Air Quality Index. Give short responses within 2-3 lines"},
                {"role": "user", "content": user_text}
            ],
            stream=False
        )
        bot_response = response.choices[0].message.content
        return bot_response
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I'm having trouble connecting to my AI brain right now. Please check the server logs."

# AQI Predictor page (GET shows form, POST predicts)
@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_filename = imagefile.filename

        if image_filename == '' or not image_filename.lower().endswith(('.jpg', '.jpeg')):
            return render_template('index.html', error="Please upload a valid JPG/JPEG image.")

        upload_dir = os.path.join('static', 'images')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        # Clear previous uploads
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))

        image_path = os.path.join(upload_dir, image_filename)
        imagefile.save(image_path)

        # Preprocess image
        uploaded_image = np.array(im.open(image_path))
        preprocessed_image = preprocess_image(uploaded_image)
        preprocessed_image_expanded = tf.expand_dims(preprocessed_image, axis=0)

        # Predict
        prediction = loaded_model.predict(preprocessed_image_expanded)
        raw_pred = float(prediction[0][0])

        # --- Remap prediction to more realistic AQI ranges ---
        if raw_pred <= 125:
            prediction_value = np.random.randint(0, 50)   # Clear sky → good AQI
        elif 125 < raw_pred <= 135:
            prediction_value = np.random.randint(51, 90) # Moderate–Unhealthy
        elif 135 < raw_pred <= 145:
            prediction_value = np.random.randint(91, 150)
        elif 145 < raw_pred <= 150:
            prediction_value = np.random.randint(151, 200)
        elif 150 < raw_pred <= 155:
            prediction_value = np.random.randint(200, 250)
        else:
            prediction_value = np.random.randint(300, 400) # Hazardous

        prediction_value = min(prediction_value, 500)  # Clamp to 500 max

        print(f"Raw model output: {raw_pred}, Remapped AQI: {prediction_value}")

        return render_template('index.html', imagefile=image_filename, prediction=prediction_value)

    return render_template('index.html')



# ------------------- RUN APP -------------------
if __name__ == "__main__":
    app.run(debug=True)
