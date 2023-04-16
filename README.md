# Tweet Predictor

The problem at hand is to develop a robust classification model that can accurately differentiate between real and fake disaster-related tweets. With the rise of social media usage during times of crisis, the spread of misinformation and fake news has become a growing concern. This can lead to confusion, panic, and even impede emergency response efforts. Therefore, it is crucial to develop a machine learning-based solution that can identify and filter out fake disaster tweets from real ones. The classification model should be able to handle large volumes of data, account for variations in language, and be able to generalize to different types of disasters. The solution will be beneficial to emergency responders, news agencies, and the public in making informed decisions during a crisis.







Dataset Description
This dataset contains tweets related to disasters and their classification as real or fake. The dataset has two parts: a training set and a test set. The training set contains 7503 rows, while the test set contains 3243 rows.

Each row in the dataset has the following columns:

- id: A unique identifier for each tweet
- text: The text of the tweet
- location: The location the tweet was sent from (may be blank)
- keyword: A particular keyword from the tweet (may be blank)
- target: In train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)
The aim of this dataset is to predict whether a tweet is about a real disaster or not. This is a classification task.

Dataset Source
This dataset is sourced from Kaggle, and can be found at the following link: https://www.kaggle.com/c/nlp-getting-started/data

Data Preprocessing
Before using this dataset for machine learning models, it may require some preprocessing. For example, the location and keyword columns may require cleaning and normalization. Additionally, the text column may require preprocessing steps such as tokenization, stemming, and stopword removal.

![location](/Users/krishnadevabhaktuni/Desktop/GA/Capstone/images/location.png)


Model Building
The goal of this dataset is to build a machine learning model to predict whether a tweet is about a real disaster or not. Various classification algorithms can be used, such as logistic regression, Naive Bayes Classifier, Random Forest Classifier. Hyperparameter tuning is performed using techniques like Gridsearch and Randomsearch. Additionally, natural language processing techniques can be used to extract features from the text column. The accuracy of the model can be evaluated using metrics such as precision, recall, and F1-score.






