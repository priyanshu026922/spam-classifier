import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
     
    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title("EMAIL SPAM OR NOT SPAM CLASSIFIER")

input_sms=st.text_input("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        tranfomed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([tranfomed_sms])
        res = model.predict(vector_input)[0]

        if res == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
#1.prerprocess
#2.vectorize
#3.predict