import pandas as pd

#%% import twitter data

data = pd.read_csv(r"gender_classifier.csv",encoding = "latin1")
data = pd.concat([data.gender,data.description],axis =1)
#concat = birleştir

data.dropna(axis = 0 ,inplace = True)
#nan olan verileri kaldırdık

data.gender = [1 if each == "female" else 0 for each in data.gender]
#"female"(kadın) =>1 male(erkek)=>0

#%% cleaning data
#regular expression = özel textler

import re


firs_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",firs_description)
#^a-zA-Z = Küçük a'dan küçük z ,A-Z kadar olanları bulma
# a-zA-Z = a-z ,A-Z olanları bul 
#"[^a-zA-Z]"," ",firs_description , ," ", boşluk ile değiştir

description = description.lower()
#Bütün verileri küçük harf yap

#%% stopwords(irrelavent words) gereksiz kelimeler
import nltk #notural language tool kit

nltk.download("stopwords")#gereksiz kelimeleri indiriyor,corpus diye bir klasöre indiriliyor

from nltk.corpus import stopwords#corpus klasöründen import ediliyor

description = description.split()
# #Kelime kelime ayır ve bir listeye at

#split yerine tokebizer kullanabiliriz
#description = nltk.word_tokenize(description)
#shouldn't ="should" , "not" olarak ayırır.

#%%Gereksiz kelimeler çıkar
description = [word for word in description if not word in set(stopwords.words("english"))]
#Gereksiz kelimeleri çıkardık




#%%LEMATAZATION ,  gitmeyeceğim =>git

import nltk as nlp

lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
#listedeki kelimelerin köklerini bul ve ata

description=" ".join(description)#Bir cümle oluştur

#%%Cleaning Data
description_list=[]
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    #^a-zA-Z = Küçük a'dan küçük z ,A-Z kadar olanları bulma
    # a-zA-Z = a-z ,A-Z olanları bul 
    #"[^a-zA-Z]"," ",firs_description , ," ", boşluk ile değiştir

    description = description.lower()
    #Bütün verileri küçük harf yap
    
    description = description.split()
    #Kelime kelime ayır ve bir listeye at
    
    description = [word for word in description if not word in set(stopwords.words("english"))]
    #Gereksiz kelimeleri çıkardık
    
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    #listedeki kelimelerin köklerini bul ve ata
    
    description_list.append(description)
    
    































