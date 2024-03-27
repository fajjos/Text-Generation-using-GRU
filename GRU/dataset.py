import nltk

# Ensure that the Penn Treebank dataset is downloaded
nltk.download('treebank')

# Import the corpus
from nltk.corpus import treebank

# Get a specific amount of text data
text_data = ' '.join(treebank.words()[:1000])

# Define the path to the output text file
output_file_path = 'penn_treebank_subset.txt'

# Save the text data to the output text file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(text_data)

print(f"Saved the text data to '{output_file_path}'")
