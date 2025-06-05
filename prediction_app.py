from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack, csr_matrix
import re
import textstat
import nltk
import joblib
import streamlit as st

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def reg_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'Ã[\x80-\xBF]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

wnl = WordNetLemmatizer()

def lemmatize_tokens(cleaned_tokens):
    lemmatized_list = []
    #Loop through the tokens in each column and lemmatized the word based on verb
    for word in cleaned_tokens:
        lemmatized_word = wnl.lemmatize(word, pos='v')
        lemmatized_list.append(lemmatized_word)
    return lemmatized_list

model = joblib.load('svm_model_fake_news_prediction.pkl')
scaler = joblib.load('scaler.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Fake News Detection Deployment") 
user_input = st.text_area("Enter news: ", height=400)


if st.button("Analyze"):

    #Get the readability score 
    readability = textstat.flesch_reading_ease(str(user_input))

    #Clean text
    cleaned_input = reg_text(user_input) 

    #Tokenized text
    tokenized_input = word_tokenize(cleaned_input)

    #Stopword removal
    custom_stopwords = {'reuters', 'donald', 'trump', 'unite', 'state', 'say', 'call', 'one'}

    stopwords_list = stopwords.words("english") + list(custom_stopwords)
    cleaned_token = [word for word in tokenized_input if word not in stopwords_list]

    #Lemmatization
    lemmatized_token = lemmatize_tokens(cleaned_token)

    #TF-IDF Vectorization
    tfidf_token = ' '.join(lemmatized_token)
    final_input = tfidf_vectorizer.transform([tfidf_token])

    #Normalize the readability score
    readability_scaled = scaler.transform([[readability]])

    #Combine readability score and input news 
    readability_sparse = csr_matrix(readability_scaled)
    input_with_score = hstack([final_input, readability_sparse])

    #Actual prediction
    prediction = model.predict(input_with_score)

    if prediction == 1:
         st.write('Legitimate News!')
    else:
         st.write('Fake News!')




