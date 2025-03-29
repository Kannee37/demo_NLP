from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import numpy as np
import pickle


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')


model_test = load_model(r'd:\mine\HocTap\nlp\model\model_lstm.h5')

with open(r'd:\mine\HocTap\nlp\model\tfidf_vectorizer.pkl', "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(r'd:\mine\HocTap\nlp\model\tokenizer.pkl', "rb") as f:
    tokenizer = pickle.load(f)

    # Hàm tiền xử lý email
def preprocess_text(text):
    # Khởi tạo stop words & lemmatizer bên trong hàm
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'\W+', ' ', text)  # Loại bỏ ký tự đặc biệt
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stop_words]  # Loại bỏ stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return " ".join(words)


def predict_email(input_text):
    max_len = 100

    #Tiền xử lý văn bản
    clean_text = preprocess_text(input_text)

    input_vector_tfidf = tfidf_vectorizer.transform([clean_text])

    # Chuyển đổi văn bản thành chuỗi số cho LSTM
    input_sequence = tokenizer.texts_to_sequences([clean_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_len, padding='post', truncating='post')


    # Dự đoán bằng LSTM
    lstm_proba = model_test.predict(input_padded)[0][0]
    lstm_pred = 1 if lstm_proba > 0.5 else 0

    #Trả về kết quả dự đoán
    if (lstm_pred == 1):
        result = 'This is a spam email!'
    else:
        result = 'This is not spam email!'
    return result