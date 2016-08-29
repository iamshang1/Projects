# Text Generation

This exercise uses a long short term memory recurrent network to generate plausible text segments by learning patterns
from an existing text document. For this exercise, Douglas Adams' Hitchhiker's Guide to the Galaxy series is used as the 
text corpus for training. To reduce the complexity of the LSTM network, the network is fed a string of characters and asked
to predict the following character instead of fed a string of words and asked to predict the following word. This
reduces the input and output vectors from over 10,000 words to only 72 characters.

Below are details about how the LSTM Recurrent Net is implemented:
 - Text is broken down by character into a categorical vector of length 72 (lowercase letters, uppercase letters, punctuation, and symbols)
 - Input vector is fed into a hidden layer of 256 LSTM nodes
 - LSTM output fed into a softmax layer that predicts the following character
 - Orthogonal initialization of weights
 - Categorical cross entropy cost function with Adam optimization
 - No regularization
 - Mini-batch training using 200-character segments per training iteration
 - For each mini-batch, the network receives 40 characters before it is asked to predict the following character

## Results

The following are examples of generated text segments after X training iterations:

 - 700 iterations: ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae
 - 4200 iterations: the sand and the said the sand and the said the sand and the said the sand and the said the sand and the said the
 - 36000 iterations:  seared to be a little was a small beach of the ship was a small beach of the ship was a small beach of the ship
 - 100000 iterations: the second the stars is the stars to the stars in the stars that he had been so the ship had been so the ship had been
 - 290000 iterations: started to run a computer to the computer to take a bit of a problem off the ship and the sun and the air was the sound
 - 500000 iterations: "I think the Galaxy will be a lot of things that the second man who could not be continually and the sound of the stars
 
Additional training iterations may yield additional improvements in grammar and semantics, but training was stopped for the sake of time.