import streamlit as st
from helper_methods import predict_english_to_hindi, predict_hindi_to_english

# Set the app title
st.title("Language Translation: English ↔ Hindi")

# Language direction selector
language = st.selectbox(
    "Choose a translation direction:",
    ["English to Hindi", "Hindi to English"]
)

# If English → Hindi translation selected
if language == "English to Hindi":
    # Input box for English text
    eng_txt = st.text_input("Enter English text:")

    if eng_txt:
        # Generate Hindi translation
        predicted_hindi_txt = predict_english_to_hindi(eng_txt)
        st.write("### Hindi Translation:")
        st.success(predicted_hindi_txt)

# If Hindi → English translation selected
if language == "Hindi to English":
    # Input box for Hindi text
    hindi_text = st.text_input("Enter Hindi text:")

    if hindi_text:
        # Generate English translation
        predicted_english_txt = predict_hindi_to_english(hindi_text)
        st.write("### English Translation:")
        st.success(predicted_english_txt)