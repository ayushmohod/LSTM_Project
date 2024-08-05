import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM
model=load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# create a function to predict next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0] # similar in input sequence
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] # ensure the sequence length matches max_sequnces_len-1
        #and from this value to end we take token list
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1) # get predicted index, whichever has high prob take that one
    for word,index in tokenizer.word_index.items(): # match with index in tokenzier list and return word
        if index==predicted_word_index:
            return word
    return None


# streamlit app
st.title("Next word prediction with LSTM")
input_text=st.text_input("Enter the sequnce of words","Haue you had quiet")
if st.button("Predict the Next Word"):
    max_sequence_len=model.input_shape[1]+1 # take same sequence we tooke to train the model
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next Word: {next_word}')