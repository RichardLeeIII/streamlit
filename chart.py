import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("data/sample.csv")

st.line_chart(df, x="year",y=["col1","col2","col3"])

st.area_chart(df, x="year",y=["col1","col2"])

st.bar_chart(df, x="year",y=["col1","col2"])

df2=pd.read_csv("data/sample_map.csv")

st.map(df2)

fig, ax = plt.subplots()
ax.plot(df.year, df.col1)
ax.set_title("my figure tiltle")
ax.set_xlabel("x label")
ax.set_ylabel("y label")
fig.autofmt_xdate()

st.pyplot(fig)