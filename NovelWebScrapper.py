import requests
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

url='https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'
r=requests.get(url)
html=r.text

soup=BeautifulSoup(html,'html5lib')
text = soup.get_text()

tokenizer= RegexpTokenizer('\w+')

tokens= tokenizer.tokenize(text)

words = []

for word in tokens:
    words.append(word.lower())

sw=nltk.corpus.stopwords.words('english')

words_ns =[]
for word in words:
    if word not in sw:
        words_ns.append(word)

sns.set()

freqdist1=nltk.FreDist(words_ns)
freqdist1.plot(25)
