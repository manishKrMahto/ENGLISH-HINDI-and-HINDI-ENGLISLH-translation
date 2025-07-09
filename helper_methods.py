import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict_english_to_hindi(eng_txt: str) -> str:
    """
    Translate English sentence to Hindi using a trained seq2seq model.

    Args:
        eng_txt (str): English sentence to translate.

    Returns:
        str: Translated Hindi sentence (whitespace trimmed).
    """
    # Add start and end tokens
    eng_txt = "<sos> " + eng_txt + " <eos>"

    # Load the trained English-to-Hindi translation model
    model = load_model("models/english_to_hindi_model.keras")

    # Load tokenizers and sequence length info
    en_tokenizer, hi_tokenizer, max_en_len, max_hi_len = pickle.load(
        open("models/tokenizers_for_english_to_hindi.pkl", "rb")
    )

    # Convert input sentence to padded sequence
    input_seq = en_tokenizer.texts_to_sequences([eng_txt])
    input_seq = pad_sequences(input_seq, maxlen=max_en_len, padding='post')

    # Start decoder with <sos> token
    decoder_input_tokens = [hi_tokenizer.word_index['<sos>']]

    decoded_sentence = ''

    for _ in range(max_hi_len):
        # Prepare decoder input
        decoder_input = np.array(decoder_input_tokens).reshape(1, -1)

        # Predict next token
        output_tokens = model.predict([input_seq, decoder_input], verbose=0)
        predicted_token = np.argmax(output_tokens[0, -1, :])

        # Convert token to word
        predicted_word = hi_tokenizer.index_word.get(predicted_token, '')

        if predicted_word == '<eos>':
            break

        decoded_sentence += ' ' + predicted_word

        # Update decoder input with new token
        decoder_input_tokens.append(predicted_token)

    return decoded_sentence.strip()


def predict_hindi_to_english(hindi_text: str) -> str:
    """
    Translate Hindi sentence to English using a trained seq2seq model.

    Args:
        hindi_text (str): Hindi sentence to translate.

    Returns:
        str: Translated English sentence (whitespace trimmed).
    """
    # Add start and end tokens
    hindi_text = "<sos> " + hindi_text + " <eos>"

    # Load the trained Hindi-to-English translation model
    model = load_model("models/hindi_to_english_model.keras")

    # Load tokenizers and sequence length info
    en_tokenizer, hi_tokenizer, max_en_len, max_hi_len = pickle.load(
        open("models/tokenizers_for_hindi_to_english.pkl", "rb")
    )

    # Convert input sentence to padded sequence
    input_seq = hi_tokenizer.texts_to_sequences([hindi_text])
    input_seq = pad_sequences(input_seq, maxlen=max_hi_len, padding='post')

    # Start decoder with <sos> token
    decoder_input_tokens = [en_tokenizer.word_index['<sos>']]

    decoded_sentence = ''

    for _ in range(max_en_len):
        decoder_input = np.array(decoder_input_tokens).reshape(1, -1)

        # Predict next token
        output_tokens = model.predict([input_seq, decoder_input], verbose=0)
        predicted_token = np.argmax(output_tokens[0, -1, :])

        predicted_word = en_tokenizer.index_word.get(predicted_token, '')

        if predicted_word == '<eos>':
            break

        decoded_sentence += ' ' + predicted_word

        decoder_input_tokens.append(predicted_token)

    return decoded_sentence.strip()