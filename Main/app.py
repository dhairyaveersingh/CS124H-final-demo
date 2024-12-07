import streamlit as st 
import altair as alt
import plotly.express as px 
import pandas as pd 
import numpy as np 
from datetime import datetime
from transformers import pipeline

# Loading pre-trained emotion classifier pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-roberta-large", top_k=None)

from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table

def predict_emotions(docx):
    results = emotion_classifier(docx)
    results_sorted = sorted(results[0], key=lambda x: x['score'], reverse=True)
    return results_sorted[0]['label']

def get_prediction_proba(docx):
    results = emotion_classifier(docx)
    return {result['label']: result['score'] for result in results[0]}


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://png.pngtree.com/background/20210709/original/pngtree-simple-technology-business-line-picture-image_938206.jpg");
             background-size: cover;
         }}
         /* General body styling */
         body {{
             font-family: 'Arial', sans-serif;
         }}
         /* Sidebar styling */
         [data-testid="stSidebar"] {{
             background: linear-gradient(180deg, #52079A, #062879); /* Gradient from dark blue to orange */
             color: white;
         }}
         [data-testid="stSidebar"] .css-1d391kg {{
             color: white;
         }}
         /* Title and headers */
         h1, h2, h3 {{
             color: #FFFFFF; /* White */
         }}
         /* Custom button style */
         .stButton button {{
             background-color: #004080; /* Dark Blue */
             color: white;
             border-radius: 8px;
             border: none;
             font-size: 16px;
             padding: 10px 20px;
             cursor: pointer;
         }}
         .stButton button:hover {{
             background-color: #FFA500; /* Orange */
         }}
         /* DataFrame styling */
         .css-17z80pu {{
             background-color: #d3d3d3; /* Grey */
             border: 1px solid #ddd;
             border-radius: 4px;
             padding: 10px;
         }}
         /* Custom chart area */
         .stAltairChart {{
             background-color: #d3d3d3; /* Grey */
             border: 1px solid #ddd;
             border-radius: 5px;
             padding: 10px;
         }}
         /* Text area styling */
         .css-91z34k {{
             background-color: #e0e0e0; /* Light Grey for Text Area Box */
             border: 1px solid #ddd;
             border-radius: 4px;
             padding: 10px;
         }}
         /* Top bar styling */
         header[data-testid="stHeader"] {{
             background: rgba(0, 0, 0, 0); /* Transparent */
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happiness":"🤗", "joy":"😂", "neutral":"😐", "sadness":"😔", "surprise":"😮"}

def main():
    st.set_page_config(page_title="Emotion Classifier App: Veer", layout="wide")
    set_bg_hack_url()

    st.sidebar.title("Menu")
    menu = ["🏠 Home", "📊 Monitor", "ℹ️ About"]
    choice = st.sidebar.selectbox("Select an Option", menu)
    

    create_page_visited_table()
    create_emotionclf_table()
    
    if choice == "🏠 Home":
        add_page_visited_details("Home", datetime.now())
        st.title("Emotion Classifier App")
        st.subheader("Enter text to analyze its emotion")
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            
            add_prediction_details(raw_text, prediction, max(probability.values()), datetime.now())

            col1, col2 = st.columns(2)

            with col1:
                st.success("Input Text")
                st.write(raw_text)

                st.success("Sentiment Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {max(probability.values()):.2f}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(list(probability.items()), columns=["emotions", "probability"])

                fig = alt.Chart(proba_df).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "📊 Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.title("App Monitoring")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename','Time_of_Visit'])
            st.dataframe(page_visited_details)    

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
            st.altair_chart(c, use_container_width=True)    

            p = px.pie(pg_count, values='Counts', names='Pagename')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):  #initially showed Unicode decode error: utf-8 codec cant decode byte; fix:
            try:
                prediction_details = view_all_prediction_details()
                df_emotions = pd.DataFrame(prediction_details, columns=['Rawtext','Prediction','Probability','Time_of_Visit'])

                # fix for unicodedecodeerror: Ensuring all columns are converted to strings to avoid decoding errors.
                df_emotions = df_emotions.applymap(lambda x: x.decode('utf-8', 'ignore') if isinstance(x, bytes) else str(x))
                st.dataframe(df_emotions)

                prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
                pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
                st.altair_chart(pc, use_container_width=True)
            except UnicodeDecodeError as e:
                st.error(f"Error decoding data: {e}")
            
    else:
        st.title("About")
        add_page_visited_details("About", datetime.now())
        st.subheader("Emotion Classifier App")
        st.text("A simple application to classify emotions from text.")

if __name__ == '__main__':
    main()
