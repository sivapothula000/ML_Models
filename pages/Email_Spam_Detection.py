import streamlit as st
import pickle
from gtts import gTTS

# Load model
model = pickle.load(open('models\spam.pkl', 'rb'))
cv = pickle.load(open('models\vectorizer.pkl', 'rb'))

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    audio_file = open("output.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

st.title("Email / SMS Spam Detection")

st.write("Enter your message below")

msg = st.text_input("Enter a text")

if st.button("Process"):
    if msg.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = cv.transform([msg]).toarray()
        result = model.predict(vec)

        if result[0] == 0:
            output = "This is Not Spam"
            st.success(output)
        else:
            output = "This is Spam"
            st.error(output)

        speak(output)