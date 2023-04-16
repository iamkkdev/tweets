import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.title("Tweet Predictor")
st.subheader("Is it a Real Disaster or Fake?")
df = pd.read_csv('./data/train.csv')
st.write('Use the sidebar to select a page to view.')

page = st.sidebar.selectbox(
    'Page',
    ('About', 'EDA', 'Make a prediction')
)

if page == 'About':
    st.subheader('About this project')
    st.markdown(''' 
             
This is a Streamlit app that hosts the Prediction of Real and Fake Disaster Tweets.
The Data was collected from Kaggle and only two columns text and target were used.
The data was cleaned using regular expressions and then lemmatized.
Lemmatization considers the context and converts the word to its meaningful base form, 
which is called Lemma.

The best model I found was....

You can get in touch with me on these websites....


    ''')
    st.dataframe(df)
    if st.button("Drop columns"):
    # Drop the 'keyword' and 'location' columns
        df = df.drop(columns=['keyword', 'location'])
        
        # Display the modified dataframe
        st.write("Modified Dataframe")
        st.dataframe(df)

elif page == 'EDA':
    # header
    st.subheader('Exploratory Data Analysis')
    st.write('''The model is trained on Tweets.
    
Below you can find the plots''')
    # Create a DataFrame with image URLs
    image_list = ['./images/location.png' ,'./images/distplot1.png', './images/kdeplot.png', 
                  './images/real.png', './images/real1.png', 
                  './images/jointgrid.png']
    #col1, col2, col3, col4, col5, col6 = st.columns(6)
    
# Display images in 3 rows with 2 columns each
    col1, col2 = st.columns(2)
    with col1:
        #st.write('## Row 1')
        image1 = Image.open(image_list[0])
        st.image(image1, caption='locations', use_column_width= True)
        image2 = Image.open(image_list[1])
        st.image(image2, caption='Dist Plot', use_column_width= True)
    with col2:
        #st.write('## Row 1')
        image3 = Image.open(image_list[2])
        st.image(image3, caption='KDE', use_column_width= True)
        image4 = Image.open(image_list[3])
        st.image(image4, caption='Frequent Words', use_column_width= True)
    with st.container():
        #st.write('## Row 3')
        col3, col4 = st.columns(2)
        with col3:
            image5 = Image.open(image_list[4])
            st.image(image5, caption='Frequent Words', use_column_width= True)
        with col4:
            image6 = Image.open(image_list[5])
            st.image(image6, caption='Jointgrid', use_column_width= True)
   
        

elif page == 'Make a prediction':
    

# Read in my model
    with open('models/logistic_tfidf.pkl', 'rb') as f:
        model = pickle.load(f)

    text = st.text_area('Please enter the text:', max_chars = 1000)
    #Predictions
    predicted_tweet = model.predict([text])[0]
    probs = list(model.predict_proba([text])[0])
    if st.button("Submit"):
            if len(text) > 0:

            # Add some pictures?
        
                if predicted_tweet == 0:
                    prob = probs[0]
                    st.write(f'The Prediction is: "It is not a Real Disaster" with a {round(100 * prob, 2)}% probability!')
                    st.image('https://media.tenor.com/UIOAoI_h-XsAAAAd/sleep-tom-and-jerry.gif')
                    

                if predicted_tweet == 1:
                    prob = probs[1]
                    st.write(f'The Prediction is: "It is a Real Disaster" with a {round(100 * prob, 2)}% probability!')
                    st.image('https://media.tenor.com/wbU069eOtu8AAAAC/hulk-smash.gif')
                    st.warning('You need to Run', icon="⚠️")
            else:
                st.error('Please write some text to generate a prediction!')


        