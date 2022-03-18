# Resume-Recommender
Project for the Booz CMD Data Science Cohort.

### Notebook usage
The main notebook is [`project.ipynb`](https://github.com/MaraudingAvenger/Resume-Recommender/blob/master/project.ipynb). I've included all the output with the notebook in the repo as well. 

### Console usage

There is a command-line version included in the [`/Console`](https://github.com/MaraudingAvenger/Resume-Recommender/tree/master/Console) folder, specifically [`document.py`](https://github.com/MaraudingAvenger/Resume-Recommender/blob/master/Console/document.py). For the console file to work correctly, it needs to be in the same directory as the extracted resume data set. 

```python
$> python document.py path/to/document.docx
```

Due to time constraints, it only works on `.docx` files. It can easily be tweaked to work on other kinds of text files by adding a parser function that runs the `clean()` and `tokenize()` functions on the text of whatever file type you want it to work on.
