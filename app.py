from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import string
import re
import pickle

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# load the model from disk
filename = 'Sentiment_analysis.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transformer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        df = pd.DataFrame(data, columns=['tweet'])
        # removing the emojis
        def remove_emojis(tweet):
            emoji = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", re.UNICODE)
            return re.sub(emoji, '', tweet)
        df['tweet'] = df['tweet'].apply(remove_emojis)

        #cleaning the text
        def cleantext(text):

            text = re.sub(r"http\S+", "", text)       # removing the hyperlink
            text= re.sub("@[A-Za-z0-9_]+","", text)   # removing the @user
            text = re.sub("#[A-Za-z0-9_]+","", text)  # removing the #user
            text = re.sub("^\\s+|\\s+$", "", text)    # removing leading and trailing white spaces
            text = re.sub("[^a-zA-Z]", " ", text)     # removing the non english words
            text = re.sub(r'(.)1+', r'1', text)       # removing the repeated tweet
            text = re.sub('[0-9]+', '', text)         # removing the numbers
            text = re.sub("\n", " ", text)            # removing the new line
            text = re.sub("[ \t]{2,}", " ", text)     # removing the two blank space
            return text
        df['tweet'] = df['tweet'].apply(cleantext)

        # removing small frequent words
        df['tweet'] = df['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

        # removing the punctuations
        english_punctuations = string.punctuation
        def cleaning_punctuations(text):
            translator = str.maketrans('', '', english_punctuations)
            return text.translate(translator)
        df['tweet']= df['tweet'].apply(lambda x: cleaning_punctuations(x))

        # taking the list of stopwords

        nltk.download('stopwords')
        stopwordlist = stopwords.words('english')
        new_stopwords = ['amazonsupport', 'amazon', 'amazonhelp', 'AmazonSupport','Amazon', 'help', 'Help',
                         'flipkartsupport', 'flipkart', 'flipkarthelp', 'Flipkart', 'FlipkartSupport', 'FlipkartHelp',
                         'snapdealsupport', 'snapdeal', 'snapdealhelp', 'SnapDeal', 'SnapDealSupport', 'SnapDealHelp',
                         'lenskartsupport', 'lenskart', 'lenskarthelp', 'LensKart', 'LensKartSupport', 'LensKartHelp',
                         'zomatosupport', 'zomato', 'zomatohelp', 'Zomato', 'ZomatoSupport', 'ZomatotHelp',
                         'bigbasketsupport', 'bigbasket', 'bigbaskethelp', 'BiBasket', 'BigBasketSupport', 'BigBasketHelp',
                         'myntrasupport', 'myntra', 'myntrahelp', 'Myntra', 'MyntraSupport', 'MyntraHelp', 'support',
                         'bigbazaar', 'bigbazaarhelp', 'BigBazaarHelp', 'customer', 'BibBazaarSupport', 'BIGBAZAAR', 'AMAZON',
                         'FLIPKARAT', 'ZOMATO', 'LENSKART', 'BIGBASKET', 'MYNTRA', 'order']

        # Adding to the list of words
        stopwordlist.extend(new_stopwords)
        # Removing the stopwords
        STOPWORDS = set(stopwordlist)
        def cleaning_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])
        df['tweet'] = df['tweet'].apply(lambda text: cleaning_stopwords(text))

        # function to convert nltk tag to wordnet tag
        lemmatizer = WordNetLemmatizer()

        # Finds the part of speech tag
        def nltk_tag_to_wordnet_tag(nltk_tag):
            if nltk_tag.startswith('J'):
                return wordnet.ADJ
            elif nltk_tag.startswith('V'):
                return wordnet.VERB
            elif nltk_tag.startswith('N'):
                return wordnet.NOUN
            elif nltk_tag.startswith('R'):
                return wordnet.ADV
            else:
                return None

            # lemmatize sentence using pos tag
        def lemmatize_sentence(sentence):

            #tokenize the sentence and find the POS tag for each token
            nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

            #tuple of (token, wordnet_tag)
            wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
            lemmatized_sentence = []
            for word, tag in wordnet_tagged:
                if tag is None:

                    #if there is no available tag, append the token as is
                    lemmatized_sentence.append(word)
                else:

                    #else use the tag to lemmatize the token
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            return " ".join(lemmatized_sentence)

        df['tweet'] = df['tweet'].apply(lambda x: lemmatize_sentence(x))

        # count vectorizer
        vect = cv.transform(df['tweet'])
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)