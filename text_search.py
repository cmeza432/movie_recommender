import glob, os, nltk, re, heapq, tarfile
import numpy as np
from nltk.corpus import stopwords
from string import punctuation

stopword = stopwords.words('english')

# Read the file and construct for better searching
def read_line(file):
    with open(file, 'rt') as fd:
        line = fd.readline()
        # For better matching, lowercase all letters
        line = np.char.lower(line)
        # Convert back into a string
        line = str(line)
        # Subsitute all single characters into no space
        line = re.sub(r'\b\w{1,1}\b', '', line)
        # Remove all break symbols from the dataset
        line = line.replace('<br />', ' ')
        # Remove numbers
        line = ''.join(c for c in line if not c.isdigit())
        line = ''.join(c for c in line if c not in punctuation)
        line = nltk.word_tokenize(line)
        result = [word for word in line if word not in stopword]
    return result

# Read the original file and leave it as original for returning review
def read_original(file):
    with open(file, 'rt') as fd:
        line = fd.readline()
        # Remove all break symbols from the dataset
        line = line.replace('<br />', '')
    return line

# Return the tf of the user query
def user_tf(text, df):
    result = np.zeros((len(df)))
    final = []
    counter = 0
    for word in text.split():
        i = 0
        # Loop through df, if text word is matched to any value, get the tf
        for key in df:
            if(key == word):
                result[i] = text.split().count(word) / float(len(text.split()))
                final.append(i)
                counter = 1
            i += 1
        # If the word is not found then add a 0 to list
        if(counter == 0):
            final.append(0)
        counter = 0
    return result, final

# Compute the tf-idf of given tf and idf by multiplying them together and returning list
def compute_tf_idf(tf, idf, checker):
    result = np.zeros((len(idf), len(tf[0])))
    for x in range(len(idf)):
        if(checker == "text"):
            result.append(tf[x] * idf[x])
        else:
            for i in range(len(tf[0])):
                result[x][i] = tf[x][i] * idf[x] 
    return result

# Use the cosine similarity of both tfidf to find similarity
def similarity(tfidf, utfidf):
    np.seterr(divide='ignore', invalid='ignore')
    similar = []
    tfidf = np.transpose(tfidf)
    utfidf = np.transpose(utfidf)
    for i in range(len(tfidf)):
        result = ((np.dot(utfidf, tfidf[i])) / (np.linalg.norm(utfidf) * np.linalg.norm(tfidf[i])))
        if(result > 0):
            similar.append(result)    
        else:
            similar.append(0)
    return similar

# Return index of those high values
def get_index(data):
    data = np.asarray(data)
    final = heapq.nlargest(10, range(len(data)), data.take)
    return list(final)

# Get the review at the index for order of given words
def get_reviews(index, original, similar, name, total):
    reviews = []
    cosine = []
    names = []
    stars = []
    for x in range(len(index)):
        location = index[x]
        reviews.append(str(original[location]))
        cosine.append(round(similar[location], 6))
        names.append(name[location])
        stars.append(float(total[location][-5]))
    return reviews, cosine, names, stars

# Create and appended list of both list given
def make_appended_list(list1, list2):
    # Convert into array to be able to append the 2 2d list together
    list1 = np.asarray(list1)
    list2 = np.asarray(list2)
    list1 = np.append(list1, list2)
    # Convert back into list type
    return list(list1)

# Reads all the files and return the values of text files
def get_files():
    # Get the txt files from all the reviews available
    negative_temp = glob.glob(("Data/test/neg/" + "*.txt"))
    negative_temp = sorted(negative_temp, key=lambda name: int(name[14:-6]))
    positive_temp = glob.glob("Data/test/pos/" + "*.txt")
    positive_temp = sorted(negative_temp, key=lambda name: int(name[14:-6]))
    # Get value of the total list to retrieve stars
    total = make_appended_list(negative_temp, positive_temp)
    # Get the orginal text to return list of data
    originalpos = map(read_original, negative_temp)
    originalneg = map(read_original, positive_temp)
    # Create and original list of both positive and negative reviews
    original = make_appended_list(originalpos, originalneg)
    return total, original

# Return the DF of cached text files
def get_df(name):
    places = []
    with open(name, 'r') as f:
        for line in f:
            currentPlace = line[:-1]
            places.append(currentPlace)
    return places

# Get the tfidf of the words
def return_tfidf(tfidf, index):
    result = []
    for x in range(len(index)):
        if(index[x] != 0):
            result.append(round(sum(tfidf[index[x]]) / 10, 5))
        else:
            result.append(0)
    return result

# Main controller for getting the reviews
def controller(text):
    # Open file with names and each line will be part of list
    with open("Data/movie_names/pos_titles.txt") as f:
        names = f.readlines()
    # Load up tar file with tfidf values in it
    tar = tarfile.open("tfidf.txt.tar.gz", "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            tfidf = f.readlines()

    # Get final text word by splitting
    user = []
    for word in text.split():
        user.append(word)
    # Strip the name of all white spaces and create floats from tfidf
    names = [x.strip() for x in names]
    tfidf = [[float(n) for n in line.split()] for line in tfidf]
    
    # Return array of all files
    total, original = get_files()
    DF = get_df('df.txt')
    # Compute the tf
    utfidf, final = user_tf(text, DF)
    # Get similar matrix and find their cosine similarity
    similar = similarity(tfidf, utfidf)
    index = get_index(similar)
    final_tfidf = return_tfidf(tfidf, final)
    # Return final reviews of movies
    reviews, cosine, final_names, stars = get_reviews(index, original, similar, names, total)
    return reviews, cosine, final_names, final_tfidf, stars, user


############################## C A C H E D ##################################
# # Pre Calculate the DF for all text files and add count of each word
# DF = {}
# calculate_df(DF, data)
# get_count_df(DF)
# tf = compute_tf(DF, data)
# idf = compute_idf(DF, data)
# tfidf = compute_tf_idf(tf, idf, "no")
# with open('tfidf.txt', 'w') as f:
#         for item in tfidf:
#             print >> f, item
# df = get_df_list(DF)
#     with open('df.txt', 'w') as f:
#         for word in df:
#             f.write('%s\n' % word)
# def get_df_list(df):
#     final_df = []
#     for key in df:
#         final_df.append(key)
#     return final_df
# # Calculate the DF before for all words
# def calculate_df(DF, text):
#     for i in range(len(text)):
#         tokens = text[i]
#         for w in tokens:
#             try:
#                 DF[w].add(i)
#             except:
#                 DF[w] = {i}

# # Return the IDF matrix for the given text
# def compute_idf(DF, data):
#     idf = []
#     N = len(data)
#     for key in DF:
#         amount = 0
#         for x in range(len(data)):
#             if(data[x].count(key) > 0):
#                 amount += 1
#         value = np.log10(N / float(amount))
#         idf.append(value)
#     return idf

# # Return Term frequency for user inputted text
# def compute_tf(df, data):
#     result = np.zeros((len(df), len(data)))  
#     i = 0  
#     for key in df:
#         for x in range(len(data)):
#             result[i][x] = data[x].count(key) / float(len(data[x]))
#         i += 1
#     result = list(result)
#     return result

# # Get the count for each dictionary unique words
# def get_count_df(DF):
#     for x in DF:
#         DF[x] = len(DF[x])
#############################################################################