# Intro-to-NLP-in-python-using-NLTK
NLP or Natural Language Processing is the primary method for computers to understand and work with human language in text form. Burton's blog posts, we will go through and discuss how to go about using tools in python to start our journey into NLP.

First thing that we need to do is to install VS code and Python. I am writing these blogs under the assumption that both are working. Next you will want to open a new terminal in ES code, a Powershell terminal, and run "pip install nltk" and let python work its magic.

Next you are going to want to run the "nltk.download()" command and this is where things get a little funky. This will open a new window that may not pop up in the foreground you may need to alt tab into it, and just go ahead and say download all, you may need to select each item individually until all of them turn green, this just downloads all of the data from the NLTK library and will make all the examples easy and smooth to run through as shown in the image below. 

![plot](https://github.com/ammon502/Into-to-NLP-in-python-using-NLTK/blob/main/nltk%20download.png)

Now let's get into what natural language processing is and How to start thinking about it and learning about it. Since computers are a large series of binary switches, text and language is not something they can interpret inherently, computers do mathematical operations on text treating it as ASCII characters or a collection of characters, or we can do match and search functions on it. Now that we have downloaded the materials, we won't need to do that again, let's open up a new Python document or Markdown document and start coding. If you "import nltk" then import everything from the book with "from nltk.book import *", you'll find many different examples of raw text ready for us to manipulate however we wish. It should look something like the image below.

![plot](https://github.com/ammon502/Into-to-NLP-in-python-using-NLTK/blob/main/import%20nltk.png)

As you can see we have many different pieces of text that are all structured very differently that we can mess around with. Now that we have our texts loaded, let’s start exploring. The first thing we can do is look at the concordance of a word. This shows us every time a word appears in the text, along with the surrounding context.

Try this:
text1.concordance("monstrous")

![plot](https://github.com/ammon502/Into-to-NLP-in-python-using-NLTK/blob/main/text1%20concordance.png)

This will show you everywhere the word monstrous appears in Moby Dick, with some surrounding words to give you context. You’ll notice how the same word might be used in different ways depending on where it appears.

You can also check what other words are used in a similar context with:
text1.similar("monstrous")

This gives you words that appear in similar contexts to monstrous — a simple but powerful way to start exploring word similarity and usage.

You can also see how often each word appears using a frequency distribution:

from nltk import FreqDist

fdist1 = FreqDist(text1)

fdist1.most_common(10)

![plot](https://github.com/ammon502/Into-to-NLP-in-python-using-NLTK/blob/main/fdist1.png)'

This shows the 10 most common words in Moby Dick. Spoiler alert: it’s mostly stuff like the, and, of — very common English words.
If you want to filter these out (called stop words), we can explore that in a future post.


One of the coolest beginner-friendly tools in NLTK is the dispersion plot, which shows where specific words appear across a text:

text1.dispersion_plot(["whale", "Ahab", "sea", "ship", "death"])

This generates a plot where each dot represents the appearance of a word at a position in the text. It helps us visually track themes or character appearances across a story. The further left on the plot a tick mark is the earlier in the text or in this case the book that word appears.

![plot](https://github.com/ammon502/Into-to-NLP-in-python-using-NLTK/blob/main/simple%20dispersion%20plot.png)

And now you can see that we have started some exploratory fun things with text and seen some simple operators that we can run on text. There are many others, easy example are various comparisons, but for times sake, I wont bore you with the trivial. Now we want to get into what natural language processing is. Now that we’ve played with entire texts, let’s zoom in on the words themselves. In NLP, we usually start with a process called tokenization — that just means breaking up raw text into smaller parts (usually words or sentences). Later steps that we may get into in NLP pipelines often assign numbers to these tokens for modeling or computation.

You’ve already seen that text1 looks like a list of words. But we can also tokenize any text ourselves:

from nltk.tokenize import word_tokenize, sent_tokenize

sample_text = "Hello there! How are you doing today? This is a test."

word_tokenize(sample_text)

sent_tokenize(sample_text)


word_tokenize() breaks text into words and punctuation.

sent_tokenize() breaks it into full sentences.

![Word Tokenize](https://github.com/ammon502/Into-to-NLP-in-python-using-NLTK/blob/main/word%20tokenize.png)
![Sentence Tokenize](https://github.com/ammon502/Into-to-NLP-in-python-using-NLTK/blob/main/sentence%20tokenize.png)

This is useful because sometimes we’re not working with a nice pre-processed book like in nltk.book, but instead with our own custom text files or some of our own raw scraped data from the web, any raw text will do. Let’s say you want to analyze your own file. First, load it like this:

with open("my_text.txt") as f:
    
    raw = f.read()

Then tokenize it:

tokens = word_tokenize(raw)

Now you can do all the fun stuff like FreqDist, concordances, and vocabulary analysis on your own data. Feel free to play around with this and analyze your own text.

The next section gets into a big idea in NLP: normalization. That’s where we start cleaning the words we work with so they’re consistent. Here are some basic steps:

Lowercasing

sample_text = [w.lower() for w in sample_text]

This makes everything lowercase so that "The", "the", and "THE" are treated the same. Even with all of the raw text out there, so much of it is unclean data that methods like this are absolutely necessary to make it readable by the computer to make sure you're analyzing whatever it is correctly.

Next, we often want to filter out stopwords — common words like "and", "the", "to" that don’t carry a lot of meaning.

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in sample_text if w.isalpha() and w not in stop_words]


We also use isalpha() to remove punctuation and numbers.

These are ways to reduce words to their base form:

from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmer.stem("running")
lemmatizer.lemmatize("running", pos="v")

Stemming is a quick-and-dirty rule-based approach (it just chops endings off). It's like using a blunt pair of scissors on words. It tries to remove prefixes or suffixes using simple rules, often without checking whether the resulting root is a real word.

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem("running"))   # 'run'
print(stemmer.stem("happiness")) # 'happi'
print(stemmer.stem("relational")) # 'relat'

As you can see, happiness gets stemmed to happi, which isn't an actual English word. That's because stemmers don't care about grammar — they're fast and simple, but sometimes inaccurate. Use stemming when speed matters more than linguistic accuracy — like for quick filtering or when working with large datasets.

Lemmatization is the smarter, more linguistically informed sibling of stemming. Instead of chopping off parts of words, it uses a dictionary and part-of-speech tags to figure out the base or “lemma” of a word — and the result is always a real word.

Example using NLTK:

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))   # 'run'
print(lemmatizer.lemmatize("better", pos="a"))    # 'good'
print(lemmatizer.lemmatize("happier", pos="a"))   # 'happy'

Note: Without the correct part-of-speech (like pos="v" for verb), lemmatization may not return what you expect. It defaults to noun if not specified.
Both help group words like run, runs, running, ran under the same concept.

Different POS tags that you could use below to differentiate different forms of speech.
| POS Tag | Meaning                    | Example                        |
| ------- | -------------------------- | ------------------------------ |
| "n"   | Noun                       | "dogs" → "dog"             |
| "v"   | Verb                       | "running" → "run"          |
| "a"   | Adjective                  | "better" → "good"          |
| "r"   | Adverb                     | "more quickly" → "quickly" |
| "s"   | Adjective Satellite (rare) | (used internally by WordNet)   |

Something to keep in mind with this, if you're working with real sentences, it's not practical to label each word manually. In these cases, NLTK has other tools that automatically Label each word.
