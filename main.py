from textblob import TextBlob
import pandas as pd 
import streamlit as st
from cleantext.sklearn import CleanTransformer
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

st.header("Sentiment Analysis",)
st.write('''#### Subjectivity --> 0 to 1 \n
- Towards 0 means universal fact \n
- Towards 1 means fact may vary from person to person or time to time or place to place \n''')
with st.expander("Analyze Text"):
    text=st.text_input("Text Here: ")
    if text:
        blob=TextBlob(text)
        if round(blob.sentiment.polarity,2)>0.25:
                st.write("Postive")
                st.write("Subjective: ",round(blob.sentiment.subjectivity,2))
        elif round(blob.sentiment.polarity,2)<-0.25:
            st.write("Negative")
            st.write("Subjective: ",round(blob.sentiment.subjectivity,2))
        else:
            st.write("Neutral")
            st.write("Subjective: ",round(blob.sentiment.subjectivity,2))

        
        # st.write("Polarity: ",round(blob.sentiment.polarity,2))
        # st.write("Subjective: ",round(blob.sentiment.subjectivity,2))
    
    pre = st.text_input("Clean Text: ") 
    sw=stopwords.words('english')
    sw.remove("not")
    sw.remove("isn't")
    sw.remove("aren't")
    sw.remove("didn't")
    sw.remove("haven't")
    sw.remove("don't")
    sw.remove("hasn't")

    if pre:
        words=pre.split()
        newdoc=''
        for word in words:
            if word not in sw:
                newdoc=newdoc+word+" "
        cleaner = CleanTransformer(no_punct=True, lower=True,no_numbers=True,replace_with_number=" ",
                            no_urls=True,replace_with_url=" ",no_emails=True,replace_with_email=" ",no_digits=True,replace_with_digit=" ",no_emoji=True,no_currency_symbols=True,replace_with_currency_symbol=" ")
        cleantext=cleaner.transform([newdoc])
        st.write(cleantext[0])
st.write("In Input CSV tweets column name should be 'tweets'")
with st.expander("Analyze CSV"):
    upl=st.file_uploader("Upload File")
    def score(pre):
        blob1=TextBlob(pre)
        return blob1.sentiment.polarity
    
    def Analyze(x):
        if x>=0.5:
            return "Postive"
        elif x<= -0.5:
            return "Negative"
        else: return "Neutral"    
        

    if upl:
        df=pd.read_csv(upl)
        # del df['Unnamed : 0']
        df['score']=df['tweets'].apply(score)
        df['Analysis']=df["score"].apply(Analyze)
        st.write(df.head(10))
    
        @st.cache_data
        def convert_df(df):
            # IMPORTANT : Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
