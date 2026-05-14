import streamlit as st
import pickle
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
st.title("Spam Message Classifier")
st.write("Type a message below and check if it's Spam or Not")
msg = st.text_area("Enter message")
if st.button("Predict"):
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)
    if prediction[0] == 1:
        st.error("SPAM MESSAGE")
    else:
        st.success("NOT SPAM")