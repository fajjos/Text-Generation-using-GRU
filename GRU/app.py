from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model = tf.keras.models.load_model('text_generation_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define maximum sequence length
max_sequence_len = 20

# Define home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from the form
        seed_text = request.form['seed_text']
        num_words = int(request.form['num_words'])

        # Generate text based on user input
        generated_text = generate_text(seed_text, num_words)

        # Render the result template with generated text
        return render_template('result.html', seed_text=seed_text, generated_text=generated_text)

    # Render the home template for GET requests
    return render_template('home.html')

# Text generation function
def generate_text(seed_text, num_words):
    # Initialize the generated text
    generated_text = seed_text

    # Generate words one by one
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        output_word = tokenizer.index_word.get(predicted_index, "")
        generated_text += " " + output_word

    return generated_text

if __name__ == '__main__':
    app.run(debug=True)
