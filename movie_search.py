#!/usr/bin/python

from flask import Flask, render_template, request
from text_search import controller
import numpy as np
import re, glob, heapq, sys


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

# Returns from the home page to run the text.html page
@app.route('/text_search', methods=['POST'])
def text_search():
    return render_template("text.html")

# Routes to the text_result page after the text.html returns
@app.route('/text_result', methods=['POST'])
def text_result():
    # Get text input from text box
    text = request.form.get("text_search")
    # Remove all duplicate white spaces and tabs, all chars and lowercase everything for better searching
    " ".join(text.split())
    text = np.char.lower(text)
    text = np.char.replace(text, "'", "")
    text = str(text)
    # Subsitute all single characters into no space
    text = re.sub(r'\b\w{1,1}\b', '', text)
    # Call the controller function
    results, cosines, names, tfidf, stars, words = controller(text)
    results = results.encode('utf-8')
    return render_template('text_result.html', w_t=zip(words, tfidf), r_c_n_s=zip(results, cosines, names, stars))

# Routes to the classifier html page
@app.route('/classifier', methods=['POST'])
def classifier():
    return render_template("classifier.html")

# Return from the classifier html page
@app.route('/classifier_result', methods=['POST'])
def classifier_result():
    text = request.form.get("classifier")
    return render_template("classifier_result.html")

if __name__ == "__main__":
    app.run()