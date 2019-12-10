# **Movie Recommender**

### Deployment
Make sure to have Python Flask installed, if not already installed simply run pip install flask. Once installed, then install the MS-COCO data set here: http://cocodataset.org/#download. MS-COCO is only used for the image caption feature.

This movie recommender app uses practices from the **Data Mining** Field to use different methods for information retrieval.
The webapp will use a Text Search, Classifier and Image Captioning to return results based on user input. It will run using
Python Flask version 1.1.1. This will be done in three different iteration. First Iteration will be the text search feature.
The dataset used will be the [IMDB Reviews dataset](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset) which is 
roughly about 150MB in size and has over 50k reviews with both a test set and training set split into 25k each.

## Text Search

The text search feature code I wrote will implement different calculations such as:

* **TF(Term Frequency)**
* **IDF(Inverse Document Frequency)**
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


## Classifier

Second Feature will be a classifier feature. This classifier feature will use the user reviews about
movies and try to classify the genre based on the word used to describe the movies. I will be using the Naive 
Bayes Classifier algorithm for calculations.

First I used the unique word TF file that I created for the Text Search. Once I read that file in, I had to seperate each movie into their respective genres and the summed up word count of each genre for that word. So if I had 30 western movie reviews and they each had the word "cowboys", then my column would represent that genre and the rows are the unique words. So if you would look at the western column, then traverse down the list of words and check the cowboys row, the value would be 30. Once I had this matrix, it would be a N x M matrix:

* **N = the amount of unique words**
* **M = the amount of different genres**

The next step would be to apply the Naive Bayes algorithm to help and build an array of values using the user inputted text. When given text by the user, I would run the algorithm with the words given for each genre so the equation is:

![Naive Bayes](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e85875a7ff9e9b557eab6281cc7ff078_l3.svg)

Which using the equation for each word would then turn into:

![Simple Naive Bayes](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8171c1fe2cbd3ed62bc3f40d682c0512_l3.svg)

By calculating the value of the words inputted and each genre would give me an array of values of length = (# of genres). Then I would simply just return the highest value of this array and that index would be the genre that it would classify it with.


## Image Caption
The Image Captioning feature will prompt the user for an image to process, then will generate a caption to search for depending on the image given. No posters will be used but only screenshots for better accuracy. The MS-COCO dataset will be used for this feature and the IMDB reviews for the search once caption is generated.

Once MS-COCO is downloaded and you have python flask installed (If not then follow instruction at the top of the page) we will preprocess the images using inceptionV3. This is done by reducing the image then normalizing the image so they all have values between -1 and 1. Next will be creating a tf model that will be a 8 x 8 x 2048 sized matrix. This will then be cashed along with the preprocess step of the inceptionV3.

Next step will be to do preprocess and tokenizing all the captions. First we will get all unique words, then limit the words to the top 5k words to limit the size of memory. Once that is done then we create a word to index, and index to word mappings.

We will then create a training dataset that will be cashed, this will be done by getting the TF of the unique words. Once this is done we will squash the shape of the InceptionV3 model from 8x8x2048 to 64x2048. Then it will pass through a CNN encoder and then the RNN will predict the next word. For the training, we will extract stored files and then pass those through these ecoders. The decoder will then return prediction which will then calculate the loss. The last step is to calculate gradients and use backpropagation on the optimizer.

Once the training is done, we can now use images and apply the training to best predict the caption.
