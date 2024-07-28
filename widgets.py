import streamlit as st
import pandas as pd

#buttons
primarybtn=st.button(label="Primary",type="primary")
secondarybtn=st.button(label="secondary",type="secondary")

if primarybtn:
    st.write('hello from primary')

#checkbox
checkbox = st.checkbox('remeber me')

if checkbox:
    st.write("i will")
else:
    st.write("i wont")

#Radio buttons
df=pd.read_csv("data/sample.csv")

radio = st.radio ("choose a column", options=df.columns[1:], index=0, horizontal=False)
st.write(radio)

#selectbox
select = st.selectbox('choose a column', options=df.columns[1:], index=0)
st.write(select)

#multiselect
multiselect = st.multiselect("choose as many c",options=df.columns[1:], default=['col1'],max_selections=3)
st.write(multiselect)

#slider
slider = st.slider('pick a number', min_value=0, max_value=10, value=0, step=1)
st.write(slider)

#text input
text_input = st.text_input('what is your name', placeholder='jb')
st.write(f"your name is {text_input}")

#number input
num_input = st.number_input('pick a number', min_value=0, max_value=10,value=0)
st.write(f"your pick is {num_input}")

#text area
txtarea=st.text_area("what?",height=200,placeholder="what")