import streamlit as st

st.text("hello world yes")

st.title("title")

st.header("header")


st.subheader("subheader")

st.markdown("markdown**text**")
st.markdown("## header")
st.markdown("### header")
st.caption("caption")

st.code(
"""
import pandas as pd
pd.read_csv(my_csv_file)
"""
)

st.text("some text")

st.latex('x=2^2')

st.text("text above divider")
st.divider()
st.text('text below divider')

st.write("some text")