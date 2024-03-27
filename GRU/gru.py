import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
import pickle

# Step 1: Load the dataset
file_path = 'penn_treebank_subset.txt'  # Update file path if necessary
with open(file_path, 'r', encoding='utf-8') as file:
    text_data = file.read()

# Step 2: Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text_data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = 20  # Adjust this value based on your dataset and desired sequence length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X, labels = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Step 3: Define the GRU model
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    GRU(150, return_sequences=True),
    GRU(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X, y, batch_size=100, epochs=200, verbose=1)  # Adjust batch size and epochs as needed

# Optional Step 5: Text Generation Function with improved sampling
def generate_text_beam_search(seed_text, next_words, model, max_sequence_len, beam_width=5, temperature=0.7):
    # Initialize the list of sequences
    sequences = [[seed_text.split(), 1.0]]

    # Generate words one by one
    for _ in range(next_words):
        all_candidates = []

        # Generate candidates for each sequence
        for seq, score in sequences:
            token_list = tokenizer.texts_to_sequences([seq])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

            # Predict probabilities for the next word
            predicted_probs = model.predict(token_list, verbose=0)[0]

            # Apply temperature to the predicted probabilities
            predicted_probs = np.log(predicted_probs) / temperature
            exp_preds = np.exp(predicted_probs)
            predicted_probs = exp_preds / np.sum(exp_preds)

            # Get the top-k words based on the probabilities
            top_indices = np.argsort(predicted_probs)[-beam_width:]

            # Extend each sequence with top-k words and calculate their scores
            for index in top_indices:
                candidate_seq = seq + [tokenizer.index_word.get(index, "")]
                candidate_score = score * predicted_probs[index]
                all_candidates.append((candidate_seq, candidate_score))

        # Select the top beam_width sequences
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    # Return the sequence with the highest score
    return ' '.join(sequences[0][0])

# Example of generating text with beam search and temperature sampling
generated_text = generate_text_beam_search("This is", 20, model, max_sequence_len, beam_width=5, temperature=0.7)
print(generated_text)

# Save the model and tokenizer
model.save('text_generation_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
