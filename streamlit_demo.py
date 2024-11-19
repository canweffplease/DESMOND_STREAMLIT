import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

def download_file_from_google_drive(file_id, dest_path):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=dest_path,
                                        unzip=False)

@st.cache_data()
def load_models_and_tokenizers():
    # Your Google Drive file ids and destination paths
    model_file_id = '1GauD9-t8V8jC0aqu9iuygAk_yLz6lsX4'
    encoder_file_id = '1m3FZ7s8Pd-heVFGRzvm61mOWIO1cH3UF'
    decoder_file_id = '1t7oLbWAZWorZ3uWAfhb7XnBE8_93sGBu'
    english_tokenizer_file_id = '1GRAzVUycazeHsUH9i1Lih1mt9AHkhIau'
    french_tokenizer_file_id = '1jC9rdkL9XjH5jAu9zhdPwEiwnI8qhdlI'

    if not os.path.isfile('models/DESMOND-v0.3-GPU.h5'):
        download_file_from_google_drive(model_file_id, 'models/DESMOND-v0.3-GPU.h5')
    if not os.path.isfile('models/v0.3ed/encoder_model.h5'):
        download_file_from_google_drive(encoder_file_id, 'models/v0.3ed/encoder_model.h5')
    if not os.path.isfile('models/v0.3ed/decoder_model.h5'):
        download_file_from_google_drive(decoder_file_id, 'models/v0.3ed/decoder_model.h5')
    if not os.path.isfile('tokenizers/english_tokenizer.pkl'):
        download_file_from_google_drive(english_tokenizer_file_id, 'tokenizers/english_tokenizer.pkl')
    if not os.path.isfile('tokenizers/french_tokenizer.pkl'):
        download_file_from_google_drive(french_tokenizer_file_id, 'tokenizers/french_tokenizer.pkl')
    print(1)
    model = load_model('models/DESMOND-v0.3-GPU.h5', compile=False)
    encoder_model = load_model('models/v0.3ed/encoder_model.h5', compile=False)
    decoder_model = load_model('models/v0.3ed/decoder_model.h5', compile=False)
    print(2)
    with open('tokenizers/english_tokenizer.pkl', 'rb') as file:
        english_tokenizer = pickle.load(file)
    with open('tokenizers/french_tokenizer.pkl', 'rb') as file:
        french_tokenizer = pickle.load(file)
    print(3)
    return model, encoder_model, decoder_model, english_tokenizer, french_tokenizer

model, encoder_model, decoder_model, english_tokenizer, french_tokenizer = load_models_and_tokenizers()

max_english_sequence_length = 20  # Adjust as needed

def preprocess_input_sentence(input_sentence):
    input_seq = english_tokenizer.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_english_sequence_length, padding='post')
    return input_seq
def decode_sequence_english(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1,1))
    target_seq[0, 0] = french_tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index in french_tokenizer.index_word:
            sampled_char = french_tokenizer.index_word[sampled_token_index]
            decoded_sentence += ' ' + sampled_char
        else:
            decoded_sentence += ' <UNK>' 


        if (sampled_char == '<end>' or len(decoded_sentence) > 100): 
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

st.title('English to French Translation')

input_sentence = st.text_input("Enter an English sentence:")

if st.button('Translate'):
    if input_sentence:
        input_seq = preprocess_input_sentence(input_sentence)
        translated_sentence = decode_sequence_english(input_seq)
        st.write("Translated sentence:", translated_sentence)
    else:
        st.write("Please enter a sentence to translate.")

# streamlit run streamlit_demo.py
