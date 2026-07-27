import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="MovieIQ EDA",layout="wide")

@st.cache_data
def load_data():
    df=pd.read_csv("data/movies.csv")
    df=df[(df["budget"]>0)&(df["revenue"]>0)].copy()
    df["success"]=(df["revenue"]>df["budget"]).astype(int)
    df["genres"]=df["genres"].fillna("Unknown")
    return df

df=load_data()

st.title("🎬 MovieIQ - Exploratory Data Analysis")

st.sidebar.header("Filters")
min_vote=st.sidebar.slider("Minimum Vote Average",0.0,10.0,0.0)
genre_list=sorted(set("|".join(df["genres"]).split("|")))
selected=st.sidebar.multiselect("Genre",genre_list)

filtered=df[df["vote_average"]>=min_vote]
if selected:
    filtered=filtered[filtered["genres"].apply(lambda x:any(g in x.split("|") for g in selected))]

c1,c2,c3,c4=st.columns(4)
c1.metric("Movies",len(filtered))
c2.metric("Successful",int(filtered["success"].sum()))
c3.metric("Success Rate",f"{filtered['success'].mean()*100:.1f}%")
c4.metric("Avg Rating",f"{filtered['vote_average'].mean():.2f}")

st.subheader("Dataset Preview")
st.dataframe(filtered)

st.subheader("1. Budget vs Revenue")
fig,ax=plt.subplots()
sns.scatterplot(data=filtered,x="budget",y="revenue",hue="success",ax=ax)
st.pyplot(fig)

genre=filtered.assign(genres=filtered["genres"].str.split("|")).explode("genres")

col1,col2=st.columns(2)
with col1:
    st.subheader("2. Genre Distribution")
    fig,ax=plt.subplots(figsize=(6,4))
    genre["genres"].value_counts().plot(kind="bar",ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader("Genre Success Rate")
    fig,ax=plt.subplots(figsize=(6,4))
    genre.groupby("genres")["success"].mean().sort_values().plot(kind="barh",ax=ax)
    st.pyplot(fig)

for col,title in [("popularity","Popularity vs Success"),
                  ("runtime","Runtime vs Success"),
                  ("vote_average","Vote Average vs Success")]:
    st.subheader(title)
    fig,ax=plt.subplots()
    sns.boxplot(data=filtered,x="success",y=col,ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig,ax=plt.subplots(figsize=(7,5))
sns.heatmap(filtered[["budget","revenue","popularity","runtime","vote_average"]].corr(),annot=True,cmap="coolwarm",ax=ax)
st.pyplot(fig)

st.subheader("EDA Insights")
st.markdown("""
- Movies with larger budgets generally earn higher revenue.
- Popular movies are more likely to be successful.
- Higher vote averages are associated with successful movies.
- Runtime has a weaker relationship with success.
- Budget and revenue are strongly correlated.
""")
