from fastapi import FastAPI
import joblib
from random import choice

app = FastAPI()
model = joblib.load('chatbot_model.pkl')
vec = joblib.load('chatbot_vec.pkl')
jwb = joblib.load('chatbot_jawaban.pkl')

@app.get("/chat/{pesan}")
def chat(pesan: str):
    user_vec = vec.transform([pesan])
    intent_us = model.predict(user_vec)[0]
    return{"Jawaban": {choice(jwb[intent_us])}}
    