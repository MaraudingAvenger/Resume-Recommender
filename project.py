import re
from typing import Callable
from functools import reduce
from collections import Counter

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from docx import Document




def pipeline(*functions: Callable) -> Callable:
    'create a pipeline function that applies functions in sequence to some value x'
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def lower_strip(x:str) -> str:
    return x.lower().strip()

def sub_punc(x:str) -> str:
    return re.sub('\W', ' ', x)

def sub_space(x:str) -> str:
    return re.sub('\s+', ' ', x)

def split(x:str) -> list[str]:
    return x.split()

def stem(x: list[str]) -> list[str]:
    p = PorterStemmer()
    return [p.stem(s) for s in x]


d = Document("Luke Chambers FT Resume RES-2020-00386.docx")

clean = pipeline(
    lower_strip,
    sub_punc,
    sub_space,
    split,
    stem
)
    
lines = [
    line 
    for line in [" ".join(word
                          for word in clean(p.text)
                          if not word in stopwords.words('english'))
                for p in d.paragraphs]
    if line
]

def get_term_freq(lines: list[list[str]]) -> Counter:
    return Counter(item for line in lines for item in line)

lines