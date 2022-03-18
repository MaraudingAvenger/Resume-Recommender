import os
import sys
import re
from typing import Any, Callable
from functools import reduce, partial

import tqdm
import nltk
import gensim
import pandas as pd

from docx import Document
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def pipeline(*functions: Callable) -> Callable:
    '''
    create a callable pipeline of functions; 
    functions f, g,...n become a single callable of  n(...(g(f(x))))
    '''
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def flatten(x: list[Any]) -> list[Any]:
    return sum(map(flatten, x), []) if isinstance(x, list) else [x]


def clean(text: str) -> list[str]:
    # pipeline funcs
    def tokenize(x): return nltk.word_tokenize(x)

    # regexes
    def url_re(s): return s if s and re.match(('(https?:\/\/)?([\w\-])+\.{1}'
                                               '([a-zA-Z]{2,63})([\/\w-]*)*\/?\??'
                                               '([^#\n\r]*)?#?([^\n\r]*)'), s) is None else "URL"

    def uname_re(s): return s if s and re.match(
        r'^@\S+', s) is None else "SCREEN_NAME"
    def hashtag_re(s): return s if s and re.match(
        r'^#\S+', s) is None else "HASHTAG"

    # filters/maps
    def url(x): return map(url_re, x)
    def uname(x): return map(uname_re, x)
    def hashtag(x): return map(hashtag_re, x)
    def lower(x): return map(lambda s: s.lower(), x)
    def len_4(x): return filter(lambda wd: len(wd) > 4, x)

    f = pipeline(tokenize, lower, url, uname, hashtag, len_4)

    return list(f(text))


def lemmatize(tokens: list[str]) -> list[str]:
    # whole-token related
    def en_stop(x): return filter(
        lambda wd: wd not in stopwords.words('english'), x)

    def lemmywinks(x): return map(lambda wd: wn.morphy(wd) or wd, x)

    f = pipeline(en_stop, lemmywinks)

    return list(f(flatten(tokens)))


def get_document_words(path: str) -> list[str]:
    'get the words from a word document and return as a list of tokens'

    doc = Document(path)

    lines = [list(clean(para.text)) for para in doc.paragraphs]

    return flatten([lemmatize(line) for line in lines])


# couple of convenience print funcs
def header(text: str) -> None:
    print()
    print("#" * max((len(text) + 4), 10))
    print("#", text, "#")
    print("#" * max((len(text) + 4), 10))


def section(text: str) -> None:
    print()
    print('-' * max(len(text), 10))
    print(text)
    print('-' * max(len(text), 10))


if __name__ == "__main__":
    from pprint import pprint
    fname = sys.argv[1]

    header("Resume Dataset Analysis")

    if not os.path.exists("UpdatedResumeDataSet.csv"):
        from zipfile import ZipFile
        with ZipFile("archive.zip", 'r') as archive:
            archive.extractall()

    df = pd.read_csv("UpdatedResumeDataSet.csv")

    section('munging text data...')
    text_data = []
    for text in tqdm.tqdm(df['Resume'], ascii=False, dynamic_ncols=True):
        text_data.append(lemmatize(clean(text)))

    header("Topic Modeling")
    dictionary = gensim.corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    section('training model...')
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus,
        num_topics=15,
        id2word=dictionary,
        passes=25
    )

    section('found topics:')
    for topic in sorted(lda_model.print_topics(num_words=3), key=lambda t: t[0]):
        print("\t", topic)

    header("Classifier")

    label = LabelEncoder()
    df['cat'] = label.fit_transform(df['Category'])
    df['clean'] = [" ".join(thing) for thing in text_data]

    text = df['clean'].values
    target = df['cat'].values

    section("training model...")
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500,
    )
    word_vectorizer.fit(text)

    features = word_vectorizer.transform(text)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        random_state=5,
        test_size=0.2
    )

    model = OneVsRestClassifier(KNeighborsClassifier())
    model.fit(x_train, y_train)

    y_pred = model.predict(features)

    # section("measurements...")
    print(f'  Training Accuracy: {model.score(x_train, y_train):.2%}')
    print(f'Validation Accuracy: {model.score(x_test, y_test):.2%}')

    section("classifying provided resume...")
    doc = " ".join(get_document_words(fname))
    doc_features = word_vectorizer.transform([doc])
    result = model.predict(doc_features)
    print(label.inverse_transform(result))
