import glob, re, nltk, os
import numpy as np
from nltk.corpus import stopwords
from string import punctuation

# Function that runs the naive bayes theorem
def naive_bayes(genres, dictionary, indexes, df, text):
    result = []
    # Get value of P(H)
    ph = 1 / float(len(genres))
    # Loop through each of the genres and get a result value
    for x in range(len(genres)):
        top = 1
        bottom = 1
        amount = 0
        N = 0
        # Loop through each word and get values, no word is equal to -1
        for i in range(len(text)):
            if(indexes[i] == -1):
                pass
            else:
                # Amount of total words in that genre equal to user
                amount += dictionary[indexes[i]][x]
                # Amount of total words of all genres for each word
                N += sum(dictionary[indexes[i]])
        # Loop through each of the words and start running calculations
        for k in range(len(text)):
            # No words found represent -1
            if(indexes[i] == -1):
                pass
            else:
                if(amount == 0 or N == 0):
                    pass
                else:
                    if(dictionary[indexes[k]][x] == 0):
                        pass
                    # When words are present get the P(An.. | H) for each word on top
                    # Then get the P(An...) for the bottom portion
                    else:
                        top *= dictionary[indexes[k]][x] / float(amount)
                        bottom *= dictionary[indexes[k]][x] / float(N)
        if(bottom == 0):
            answer = 0
        else:
            # Then calculate the Top * P(H) / bottom == P(An... | H) * P(H) / P(An...)
            answer = (top * ph) / bottom
        result.append(answer)
    # once the loop is done, get the max value of that array
    best = max(result)
    # Find index that matches that max value
    answer = result.index(best)
    # Return the genre of that max value
    return genres[answer]

# returns the array index of those values found from text input in dictionary
def get_index(text, df):
    result = []
    for x in range(len(text)):
        try:
            result.append(df.index(text[x]))
        except ValueError:
            result.append(-1)
    return result

def classifier_controller(text):
    # Open file for genres given the txt file
    with open("genres.txt") as f:
        genres = f.readlines()
    genres = [x.strip() for x in genres]
    # Open the df file for word index
    with open("df.txt") as f:
        df = f.readlines()
    df = [x.strip() for x in df]
    # Load the matrix for the dictionary count of each genre
    dictionary = np.loadtxt("final_dictionary.txt")
    # Get the indexes of each word given from text
    indexes = get_index(text, df)
    # Get the resultant and classified genre for the text
    result = naive_bayes(genres, dictionary, indexes, df, text)
    return result

###################### C A C H E D ##########################
# stopword = stopwords.words('english')
# # Read the file and construct for better searching
# def read_line(file):
#     with open(file, 'rt') as fd:
#         line = fd.readline()
#         # For better matching, lowercase all letters
#         line = np.char.lower(line)
#         # Convert back into a string
#         line = str(line)
#         # Subsitute all single characters into no space
#         line = re.sub(r'\b\w{1,1}\b', '', line)
#         # Remove all break symbols from the dataset
#         line = line.replace('<br />', ' ')
#         # Remove numbers
#         line = ''.join(c for c in line if not c.isdigit())
#         line = ''.join(c for c in line if c not in punctuation)
#         line = nltk.word_tokenize(line)
#         result = [word for word in line if word not in stopword]
#     return result

# # Create and appended list of both list given
# def make_appended_list(list1, list2):
#     # Convert into array to be able to append the 2 2d list together
#     list1 = np.asarray(list1)
#     list2 = np.asarray(list2)
#     list1 = np.append(list1, list2)
#     # Convert back into list type
#     return list(list1)

# # Return the files of the given data
# def get_files():
#     # Get the txt files from all the reviews available and sort it by name
#     negative_temp = glob.glob(("Data/test/neg/" + "*.txt"))
#     negative_temp = sorted(negative_temp, key=lambda name: int(name[14:-6]))
#     positive_temp = glob.glob("Data/test/pos/" + "*.txt")
#     positive_temp = sorted(negative_temp, key=lambda name: int(name[14:-6]))
#     # Read each line and use stopwords to remove frequency
#     positive = map(read_line, positive_temp)
#     negative = map(read_line, negative_temp)
#     # Create one appended list of both positive and negative reviews
#     data = make_appended_list(positive, negative)
#     return data

# # Return the values of each word and the amount of times it occurs at each review
# # So function returns a len(DF) x len(data) matrix
# def return_dictionary(data, DF):
#     result = np.zeros((len(DF), len(data)))
#     # For each word, count the times it appears for each document
#     for i in range(len(DF)):
#         for k in range(len(data)):
#             result[i][k] = data[k].count(DF[i])
#     return result
# data = get_files()
# dictionary = return_dictionary(data, DF)
# np.savetxt("dictionary.txt", dictionary, fmt ='%.0f')

# # Create matrix of values of each word found in each genre
#     # final = get_genre_word_count(genres, dictionary, data)
# np.savetxt("final_dictionary.txt", final, fmt ='%.0f')

# Return the values of the word count for each genre for each word
# def get_genre_word_count(genres, dictionary, data):
#     # Result is going to have rows of the words, cols of unique genres
#     result = np.zeros((len(dictionary), len(data)))
#     # Loop through each word, then through genre of movies to find value
#     for i in range(len(dictionary)):
#         for k in range(len(genres)):
#             # Get the position of the genre k, then add that value of dictionary to that genre
#             position = data.index(genres[k])
#             result[i][position] += dictionary[i][k]
#     return result

# # Write out data genres to file to cache
# with open('genres.txt', 'w') as f:
#     for genre in data:
#         f.write('%s\n' % genre)

# Create list of genre and amount of words counted for each
    # data = return_unique(genres)


# # Return the DF of cached text files
# def get_df(name):
#     places = []
#     with open(name, 'r') as f:
#         for line in f:
#             currentPlace = line[:-1]
#             places.append(currentPlace)
#     return places

# # Traverse through genres and return all unique genres
# def return_unique(data):
#     unique = []
#     for x in data:
#         if x not in unique:
#             unique.append(x)
#     return unique

# # Open file with the genres for each review
# with open("Data/movie_genre.txt") as f:
#     genres = f.readlines()
# genres = [x.strip() for x in genres]
#######################################################################################