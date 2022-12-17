import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn import metrics
from skrlean.svc import LinearSVC


labels = os.listdir('../input/dataset') 
raw_data = tf.keras.preprocessing.text_dataset_from_directory(
    '../input/dataset',
    labels = "inferred",
    label_mode = "int",
    max_length = None,
    shuffle=True,
    seed=11,
    validation_split=None,
    subset=None,
)
labels_akhbaraona = os.listdir('../input/akhbarona-data') 
raw_data_akhbaraona = tf.keras.preprocessing.text_dataset_from_directory(
    '../input/akhbarona-data',
    labels = "inferred",
    label_mode = "int",
    max_length = None,
    shuffle=True,
    seed=11,
    validation_split=None,
    subset=None,
)

x=[]
y=[]
x_akh=[]
y_akh=[]
for text_batch, label_batch in raw_data:
    for i in range(len(text_batch)):
        s=text_batch.numpy()[i].decode("utf-8") 
        x.append(s)
        y.append(raw_data.class_names[label_batch.numpy()[i]])
for text_batch, label_batch in raw_data_akhbaraona:
    for i in range(len(text_batch)):
        s=text_batch.numpy()[i].decode("utf-8") 
        x_akh.append(s)
        y_akh.append(raw_data_akhbaraona.class_names[label_batch.numpy()[i]])
x = x + x_akh
y = y + y_akh

unique, counts = np.unique(y, return_counts=True) #plt y 
plt.figure("classe Pie", figsize=(10, 10))
plt.title("Classes Frequancy")
plt.bar(unique , counts,color = "maroon" , width = .4)
plt.legend(unique)
plt.show()

data = pd.DataFrame({"text":x,"label":y})


data.info()

data.drop_duplicates(inplace=True)
nltk.download('stopwords')
stop_words = list(set(stopwords.words('arabic')))

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
arabic_diacritics = re.compile("""   ّ  |   َ  |   ً  |   ُ  |   ٌ  |   ِ  |   ٍ  |   ْ  | ـ   """, re.VERBOSE)

def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', str(text))
    return text

def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    text = remove_diacritics(text)
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text




data['cleaned_text'] = data['text'].apply(clean_text)
data.head()


List = []
for row in data.itertuples():
   List.append(re.sub(r'[\W\s]', ' ', row.text))
data['attempt_million_cleaned'] = List
data.head()

label_encoder = LabelEncoder()
data['encodedLabel'] = label_encoder.fit_transform(data['label'])
data.head()

data.to_csv(r'./textClassSecond_murtadha.csv', index = False)

X_cla = data['attempt_million_cleaned'] 
y_cla = data['encodedLabel']

X_train, X_test, y_train, y_test = train_test_split(X_cla, y_cla, test_size=0.33, random_state=5)

#print('Training Data Shape:', X_train.shape)
#print('Testing Data Shape: ', X_test.shape)
#y_test.value_counts()
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train) 
X_train_tfidf.shape


clf = LinearSVC()
clf.fit(X_train_tfidf,y_train)


text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC()),])
text_clf.fit(X_train, y_train)  
predictions = text_clf.predict(X_test)
print(metrics.accuracy_score(y_test,predictions))
List_Class = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']
num_class = text_clf.predict([''])[0]
print(List_Class[num_class])




