from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import pickle

# cleaning and dropping stop words
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('wordnet')

#stemming
from nltk.stem import PorterStemmer

# lemmitizing
from nltk.stem import WordNetLemmatizer

#firebase
import pyrebase
import time
import datetime



all_stopwords = stopwords.words('english')
all_stopwords.extend(["senthilbalaji","adani","chemistrycanada","colorado","location","gaza","gwadar","implicitweet"])
all_stopwords.extend(["kotri","kotri","ptshrikant","sindh","mkstalin"])

def preprocess(s):
    #cleaning the data
    s = s.lower()
    
    # replacing ||| with space
    s = re.sub(r"\|\|\|", " ", s)
    
    s = re.sub(r"https?:\/\/(www)?.?([A-Za-z_0-9-]+)([\S])*", " ", s)

    
    # dropping emails
    s = re.sub(r"\S+@\S+", "", s)


    # dropping punctuations
    s = re.sub('[^\w\s]', " ", s)
  
    
    m = s.split(" ")
    for i in m:
        if(i in all_stopwords):
            m.remove(i)
            
    s = " ".join(m)
    if(s == ""):
        return 0
    
    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # lemmitizing (excluding stop words in this step)
    word_list = nltk.word_tokenize(s)
   
    s = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
   
    #stemming   
    ps = PorterStemmer()
    word_list = nltk.word_tokenize(s)
   
    s = ' '.join([ps.stem(w) for w in word_list])
    
    if(s == ""):
        return 0
          
    
    return s


def issue(s):
    
    data = pd.read_csv("static/Preprocessed_train_data.csv")
    X_train = data.clean_posts
    y_train = data.Issue_id
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english', ngram_range=(1,2))
    X_train = vectorizer.fit_transform(data.clean_posts)
    ch2 = SelectKBest(chi2, k=250)
    
    X_train = ch2.fit_transform(X_train, y_train)
    
    
    X_test = np.array([s])
  
    X_test = vectorizer.transform(X_test)
    X_test = ch2.transform(X_test)
    
    SVM_model = pickle.load(open("static/models/SVM.sav", 'rb'))
    result = SVM_model.predict(X_test)
   
    return result[0]

#--------------------------------------------------------------
def severe(s):
    
    data = pd.read_csv("static/Preprocessed_severity.csv")
  
    X_train = data.clean_posts
    y_train = data.Severity
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english', ngram_range=(1,3))
    X_train = vectorizer.fit_transform(data.clean_posts)
    ch2 = SelectKBest(chi2, k=35)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = np.array([s])
    X_test = vectorizer.transform(X_test)
    X_test = ch2.transform(X_test)
       
    LR_model = pickle.load(open("static/models/LR_sev.sav", 'rb'))
    
    LR_result = LR_model.predict(X_test)
       
    return LR_result[0]

def push_data(id,email,location,message,progress,issue_type, severity):
        config = {
                "apiKey": "AIzaSyDV_CPfNZ_9D_gmQrxZcbo7SMS_9GQGYcE",
                "authDomain": "city-5dc6f.firebaseapp.com",
                "databaseURL": "https://city-5dc6f-default-rtdb.firebaseio.com",
                "projectId": "city-5dc6f",
                "storageBucket": "city-5dc6f.appspot.com",
                "messagingSenderId": "876826563048",
                "appId": "1:876826563048:web:d82f14c0aad5094cf6145d"
        }
        # Initialize Firebase
        
        firebase = pyrebase.initialize_app(config)
        db = firebase.database()
        time.sleep(2)
        db.child("UserInfo").child(id).set({"email":email})
        db.child("UserInfo").child(id).update({"location":location})
        db.child("UserInfo").child(id).update({"message":message})
        db.child("UserInfo").child(id).update({"progress":progress})
        if(issue_type == 0):
            iss = "electric"
        elif(issue_type == 1):
            iss = "crime"   
        elif(issue_type == 2):
            iss = "water"   
        elif(issue_type == 3):
            iss = "sewage"   
        elif(issue_type == 4):
            iss = "road"     
        elif(issue_type == 5):
            iss = "garbage"        
                                            
        db.child("UserInfo").child(id).update({"issue":iss})
        if(int(severity) == 0):
            ss = "Not Severe"
        else:
            ss = "Severe"     

        db.child("UserInfo").child(id).update({"severity":ss})
        time.sleep(3)

        
        return "success"

app = Flask(__name__)

@app.route("/sms", methods=['POST'])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    
    # Use this data in your application logic
    from_number = request.form['From']
    body = request.form['Body']
    l = body.split(" ")
    email = l[0]
    location = l[1]
    l.pop(0)
    l.pop(0)
    message = "  ".join(l)
    print(message)
    d = datetime.datetime.now()
    tid = d.strftime('%Y%m%d%H%M%S')
    com = preprocess(message)    
    progress = 0       
    severity = severe(com) 
    issue_type = issue(com) 
    q = push_data(tid,email,location,message,progress,issue_type, severity)

    # Start our TwiML response
    resp = MessagingResponse()

    # Add a message
    resp.message("Your id is:")

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)