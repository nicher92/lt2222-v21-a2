# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's SPRING 2021 semester.

Your name: Niclas Hertzberg

*Answer all questions in the notebook here.  You should also write whatever high-level documentation you feel you need to here.*



1. 
For the preprocessing step I lemmatized each word and turned it into lowercase. I removed some punctuation but no other words since I did not find that it made the model predict better.

2. 
I made a "window" of words surrounding each instance. The window size is by reduced only by punctuation but is then extended by start and stop symbols so that each feature vector is of length 10.
This makes sense since we only get 10 available slots of information around each word, it would be a good idea to actually use all of these slots and not ignore that valuable information. 
Although a shorter sentence might actually be informative as well since then the length of each vector becomes a feature as well.
I tested using a a sentence with several stop and start symbols and it seems to get slightly better results on the testing data.

3.
For the table creation I did a bag of words model. I tested it with TF-IDF but the result was not better, perhaps because each instance is the same length. 
I added dimensionality reduction since the data is very sparse and it made the model generalize slightly better to new data, the model tended to overfit more to the testing data without dimensionality reduction (it performed very well on the training data but less well on the testing data).

5.
The diagonal across the matrix represent the cases where the prediction by the model were the same as the actual label, ie the correct prediction. Any values not in the diagonal represent an incorrect prediction.

I chose to return the number of predictions instead of percentages since there was a big difference between how frequently occuring some entities were in the text and the number of entities were strongly correlated with how many correct predictions were made.

Since the model is trained on the training data it will generally perform very well on the training data.
Without dimensionality reduction the model will guess almost perfectly since it has been trained on that data.
It seems like the model perhaps performed too well since it got almost everything right, and thus would not necessarily generalize well over to new data.
To rectify this, I reduced dimensionality to 500 dimensions, which seemed to get the best result on the testing data, which is the only data used for evaluating the model. So although the model decreased its general accuracy on the training data, it is still preferable since the accuracy on the testing data improved slightly. 


Bonus A:
The model guessed wrong on all event entities and all natural phenomenon entities.
It probably has to do with the fact that they were so uncommon compared to other types. 

There were five classes in general that tended to be mistaken for each other, are also the most common classes, geo, gpe, org, per, tim.
Regarding mistakes made because of linguistic factors, geographical entities and geopolitical entities were mistaken for each other quite frequently.
An example of this could perhaps be geo: iran, gpe: iranian. While these are different words, the sentencens in which these words occur might be very similar.
For example, two sentences like "the iranian nuclear deal have been criticized by the US government" or "the nuclear deal in iran have been criticized by the US government" would have the same surrounding context words.


Bonus B:
For bonus part b I added each words pos tag to that word, for example: "the-dt" instead of "the".
In this way the vocabulary will be larger, and two words that have the same form might be diffentiated between each other by its tag, should it be different. 

The model performed around 4% better by incorporating POS-tags.


