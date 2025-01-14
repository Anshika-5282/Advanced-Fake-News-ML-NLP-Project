import streamlit as st
import pickle

with open('vector.pkl','rb') as f:
    my_vector = pickle.load(f)

with open('best_model.pkl','rb') as f:
    my_model = pickle.load(f)

st.title('FAKE NEWS DETECTION')
st.header('ENTER THE NEWS BELOW TO CHECK WHETHER IT IS FAKE OR REAL')

news = st.text_input('Enter the news: ')
if st.button('SUBMIT'):
    if news is not None:
        vector_result = my_vector.transform([news])
        final_result = my_model.predict(vector_result)[0]
        if final_result==0:
            st.write('FAKE NEWS❌❌')
        else:
            st.write('REAL NEWS✅✅')

