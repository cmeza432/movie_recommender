# **Movie Recommender**

This movie recommender app uses practices from the **Data Mining** Field to use different methods for information retrieval.
The webapp will use a Text Search, Classifier and Image Captioning to return results based on user input. It will run using
Python Flask version 1.1.1. This will be done in three different iteration. First Iteration will be the text search feature.
The dataset used will be the [IMDB Reviews dataset](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset) which is 
roughly about 150MB in size and has over 50k reviews with both a test set and training set split into 25k each.

## Text Search

The text search feature code I wrote will implement different calculations such as:

* **TF(Term Frequency)**
* **IDF(Inverse Document Frequency)
* **TFIDF**
* **Cosine Similarity**

First using the glob feature I imported both positive and negative reviews directories. Then mapped
both into one 25k file. I used pythong directory feature to make each word in the document into a 'Key' to store the amount
that word is used in the span of the whole document to help with the calculation of the IDF. Then using the given text from
the user I parsed it into seperate dimensions of a list type. Then calculated the TF for each word of user inputted text
spanned across every document in the directory. So each word is calculated 25k times and stored into a matrix. Also found 
TF of each text spanned across the text itself treating it like a document.Once that is done then I found the IDF by using the dictionary created earlier for counting the amount of "different" words
the document contained and the amount of times that word appeared in all file. So using T(one term of user inputted text)
Next was calculating the TFIDF which was easier once TF and IDF was found. All we do from here is multiply both to get TFIDF.
Equations used for all calculations are below:

* **IDF(t) = log_e(Total number of documents / Number of documents with term t in it)**
* **TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)**
* **TFIDF = TF x IDF**

Next was getting the cosine similarity which used the TFIDF of both the user inputted text result on the span of all
documents which will be q, and the TFIDF of user text on itself as d. TFIDF is has N vectors, and user_TFIDF has one vector.
So by doing the calculation of user_TFIDF on each TFIDF vector for the amount of N documents is calculated:

![Cosine](https://wikimedia.org/api/rest_v1/media/math/render/svg/1d94e5903f7936d3c131e040ef2c51b473dd071d)

Once all similarity values are calculated, then I get the 20 highest values and their indexes. I use the indexes to help me return the value of the Names, Rating, TFIDF value for the document, Similarity of that document, and the Review.
