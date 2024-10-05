import nltk
from textblob import TextBlob
from newspaper import Article
# from analyzer import manual_testing, wordopt, vectorization, LR, gbc, rfc

nltk.download('punkt')
url = input("Enter the URL of the article: ")
article = Article(url, headers={'User-Agent': 'Mozilla/5.0'})
article.download()
article.parse()
article.nlp()


summary = article.summary
print(f'Title: {article.title}')
print(f'Summary: {summary}')
# print(manual_testing(wordopt(summary)))
