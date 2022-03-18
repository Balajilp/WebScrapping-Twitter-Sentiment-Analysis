# WEB SCRAPPING - TWITTET SENTIMENT ANALYSIS ON INDIA'S TOP ONLINE RETAILERS (2019 TO 2022)
• A glimpse of the web app:
![Screenshot (124)](https://user-images.githubusercontent.com/88379464/158972315-ef8d31d9-e21b-480e-b872-940b709b9640.png)

## Data Collection

Tweets relate to top retailers in india are collected using snscraper module using python.
Tweets are collected over a period of 2019-01-01 to 2021-12-26 using key words like 
ajio, bigbilliondays, primedaysale, bigbasket and lenskart. Codes related to the data collection 
are available in the reposetory.


## Packages Used 
1. SnScraper
2. Pandas
3. Regex 
4. Matplotlib
5. Sklearn
6. Wordcloud
7. Textblob
8. Flask 


## Data Preprocessing 

It is essential to preprocess the data before feeding to the model necessary data cleaning stets are implemented.
1. Removed "@" mentions
2. Removed "#" tags
3. Removed punctuations 
4. Removed Numerical values 
5. Removed Duplicates
6. Removed Hyperinks 
7. Removed Underscores
8. Removed Next lines 
9. Removed Emojis 
10. Converted the data in to lowecase
 
 As there was no target column, it is created by calculating the subjectivity and polarity 
 the tweets having polarity greater than 1 are classified as positive tweets,  the tweets having polarity less than 0 are classified as Negative tweets,
  the tweets having polarity equal to 0 are classified as neutral tweets, all the neutral tweets are dropped and a bar graph is plotted to show the 
  difference between positive and negative tweets. As we can't use text data to feet to a Machine Learning model we converted the text data into vectors
  using Tf-idf Vectorizer. 
  
  ## Model Building
  As we have only one target column and one independent column and the data we had was catrgorical data so i have selected Random forest and used
  converted the data in to train and test data and the data is fed to the model. The evaluation of the model are as below.
  
  |    |Precission | Recall  | F1 score   | support   | 
| ------------- | ------------- |------------- |------------- |------------- |
| 0   | 0.91  | 0.84  | 0.87  | 1720  |
| 1  | 0.93 | 096  | 0.94  | 3617|
| Accuracy  |   |   |  0.92 | 5337  |
| macro avg   |0.92 | 0.90  |   0.91 | 5337   | 
| weighted avg   |0.92 | 092  |   0.92 | 5337   |

## I have added a power point representation (PPT). For quick overview and visuazlization and solutions please go through it.
