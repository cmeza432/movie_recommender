#!/usr/bin/python

from flask import Flask, render_template, request
import numpy as np
import re
import glob
import heapq
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

# Read the file and construct for better searching
def read_line(file):
    with open(file, 'rt') as fd:
        line = fd.readline()
        # For better matching, lowercase all letters
        line = np.char.lower(line)
        # Remove all ' values
        line = np.char.replace(line, "'", "")
        # Convert back into a string
        line = str(line)
        # Subsitute all single characters into no space
        line = re.sub(r'\b\w{1,1}\b', '', line)
        # Remove all break symbols from the dataset
        line = line.replace('<br />', ' ')
    return line

# Read the original file and leave it as original for returning review
def read_original(file):
    with open(file, 'rt') as fd:
        line = fd.readline()
        # Remove all break symbols from the dataset
        line = line.replace('<br />', '')
    return line

# Calculate the DF before for all words
def calculate_df(DF, text):
    for i in range(len(text)):
        tokens = text[i]
        for w in tokens.split():
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

# Get the count for each dictionary unique words
def get_count_df(DF):
    for x in DF:
        DF[x] = len(DF[x])

# Return Term frequency for user inputted text
def compute_tf(text, data):
    user = []
    check = 0
    present = np.zeros([len(text.split()), len(data)])
    tf = np.zeros([len(text.split()), len(data)])
    for word in text.split():
        user.append(word)
        for i in range(len(data)):
            if(len(data[i].split()) != 0):
                counter = data[i].split().count(word) / float(len(data[i].split()))
                tf[check][i] = counter
                if(counter != 0):
                    present[check][i] = (1)
        check += 1
    present = list(present)
    tf = list(tf)
    return tf, user, present

# Return Term frequency for user inputted text on itself
def compute_tf_text(text, data):
    user = []
    check = 0
    present = np.zeros([len(text.split()), 1])
    tf = np.zeros([len(text.split()), 1])
    for word in text.split():
        user.append(word)
        counter = data.split().count(word) / float(len(data.split()))
        tf[check][0] = counter
        present[check][0] = (1)        
        check += 1
    present = list(present)
    tf = list(tf)
    return tf, user, present

# Return the IDF matrix for the given text
def compute_idf(text, DF, data, present):
    idf = []
    N = len(data)
    i = 0
    for word in text:
        if(word in DF.keys()):
            value = np.log10(N / float(sum(present[i])))
            idf.append(value)
        else:
            print(word)
        i += 1
    return idf

# Compute the tf-idf of given tf and idf by multiplying them together and returning list
def compute_tf_idf(tf, idf):
    result = np.zeros([len(tf), len(tf[0])])
    # Go through each word and multipy the values of their tf and idf
    for x in range(len(tf)):
        for i in range(len(tf[0])):
            result[x][i] = round(tf[x][i] * idf[x], 8)
    result = list(result)
    return result

# Use the cosine similarity of both tfidf to find similarity
def similarity(tfidf, user_tfidf):
    np.seterr(divide='ignore', invalid='ignore')
    similar = []
    tfidf = np.asarray(tfidf)
    user_tfidf = np.asarray(user_tfidf)
    for i in range(len(tfidf)):
            result = ((np.dot(user_tfidf.T, tfidf[i])) / (np.linalg.norm(user_tfidf.T) * np.linalg.norm(tfidf[i])))
            if(result > 0):
                similar.append(result)    
            else:
                similar.append(0)
    return similar

# Return index of those high values
def get_index(data):
    data = np.asarray(data)
    final = heapq.nlargest(20, range(len(data)), data.take)
    return list(final)

# Get the review at the index for order of given words
def get_reviews(index, original, tfidf, similar, name, total):
    reviews = []
    cosine = []
    table = []
    names = []
    s = []
    for x in range(len(index)):
        location = index[x]
        reviews.append(original[location])
        cosine.append(similar[location])
        table.append(tfidf[location])
        names.append(name[location])
        s.append(float(total[location][-5]))
    return reviews, cosine, table, names, s

def make_appended_list(list1, list2):
    # Convert into array to be able to append the 2 2d list together
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)
    list1 = np.append(list1, list2)
    # Convert back into list type
    return list(list1)

# Create dictionary values to return for iteration
def put(reviews, cosines, tables, names, s):
    result = []
    cosine = []
    table = []
    name = []
    stars = []
    for x in range(len(reviews)):
        result.append(reviews[x])
        cosine.append(round(cosines[x],6)) 
        table.append(tables[x])
        name.append(names[x])
        stars.append(s[x])
    return result, cosine, table, name, stars

# Main controller for getting the reviews
def controller(text):
    # Open file with names and each line will be part of list
    with open("Data/movie_names/pos_titles.txt") as f:
        names = f.readlines()
    # Get the txt files from all the reviews available
    data_temp = glob.glob(("Data/test/neg/" + "*.txt"))
    pos_temp = glob.glob("Data/test/pos/" + "*.txt")
    total = make_appended_list(data_temp, pos_temp)
    # Remove all the whitespaces for the names
    names = [x.strip() for x in names]
    # Get the files into a python 2d list
    data = map(read_line, data_temp)
    pos = map(read_line, pos_temp)
    originalpos = map(read_original, data_temp)
    originalneg = map(read_original, pos_temp)
    # Get the appended list of both reviews and original
    data = make_appended_list(pos, data)
    original = make_appended_list(originalpos, originalneg)
    # Pre Calculate the DF for all text files and add count of each word
    DF = {}
    calculate_df(DF, data)
    get_count_df(DF)
    N = len(DF)
    # Compute the tf
    tf, user, present = compute_tf(text, data)
    d, u, p = compute_tf_text(text, text)
    # Compute the idf
    idf = compute_idf(user, DF, data, present)
    text_idf = compute_idf(u, DF, data, p)
    # Compute the tf_idf matrix
    tf_idf = compute_tf_idf(tf, idf)
    text_tf_idf = compute_tf_idf(d, text_idf)
    tf_idf = np.reshape(tf_idf, (len(data), len(text_idf)))
    similar = similarity(tf_idf, text_tf_idf)
    index = get_index(similar)
    # Return final reviews of movies
    reviews, cosine, table, final_names, stars = get_reviews(index, original, tf_idf, similar, names, total)
    result, cosines, tables, tp, s = put(reviews, cosine, table, final_names, stars)
    return result, cosines, tables, tp, s, user

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/text_search', methods=['POST'])
def text_search():
    return render_template("text.html")

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
    results, cosines, tables, names, stars, words = controller(text)
    return render_template('result.html', words=words, r_c_t_n_s=zip(results, cosines, tables, names, stars))