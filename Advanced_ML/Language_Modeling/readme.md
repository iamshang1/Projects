#Language Modeling

This exercise uses a long short term memory recurrent network to learn a language model of a corpus and then generate new 
plausible text segments based on patterns from that corpus. For this exercise, Douglas Adams' Hitchhiker's Guide to the Galaxy
series is used as the text corpus for training. Two separate LSTM models are tested--the first model is a character-level model
and the second model is a word-level model.

##Model 1 - Character Level LSTM

The first LSTM model is fed a string of characters and asked to predict the following character. This reduces the input and 
output vectors from over 10,000 words to only 72 characters.

Below are details about how the LSTM Recurrent Net is implemented:
 - Text is broken down by character into a categorical vector of length 72 (lowercase letters, uppercase letters, punctuation, and symbols)
 - Input vector is fed into a hidden layer of 256 LSTM nodes
 - LSTM output fed into a softmax layer that predicts the following character
 - Orthogonal initialization of weights
 - Categorical cross entropy cost function with Adam optimization
 - No regularization
 - Mini-batch training using 200-character segments per training iteration
 - For each mini-batch, the network receives 40 characters before it is asked to predict the following character

##Model 1 Results

The following are examples of generated text segments after X training iterations:

 - 700 iterations: ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae  ae
 - 4200 iterations: the sand and the said the sand and the said the sand and the said the sand and the said the sand and the said the
 - 36000 iterations: seared to be a little was a small beach of the ship was a small beach of the ship was a small beach of the ship
 - 100000 iterations: the second the stars is the stars to the stars in the stars that he had been so the ship had been so the ship had been
 - 290000 iterations: started to run a computer to the computer to take a bit of a problem off the ship and the sun and the air was the sound
 - 500000 iterations: "I think the Galaxy will be a lot of things that the second man who could not be continually and the sound of the stars
 
Additional training iterations may yield additional improvements in grammar and semantics, but training was stopped for the sake of time.

##Model 2 - Word Level LSTM

The second LSTM model is fed an array of words converted into Word2Vec embeddings and then asked to predict the following word.

Below are details about how the LSTM Recurrent Net is implemented:
 - All words are converted into word embeddings of length 100 using Gensim's Word2Vec (all punctuation and capitalization is ignored)
 - Input vector is fed into a hidden layer of 128 LSTM nodes
 - LSTM output fed into a softmax layer that predicts the following word (approximately 11K possible words)
 - Orthogonal initialization of weights
 - Categorical cross entropy cost function with Adam optimization
 - No regularization
 - Mini-batch training using 120-word segments per training iteration
 - For each mini-batch, the network receives 20 word embeddings before it is asked to predict the following word
 
##Model 2 Results

The following are examples of generated text segments after X training iterations:

 - 1000 iterations: of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the
 - 5000 iterations: was a little a little and then he was a little a little and then he was a little a little and then he was a little a little and then he was a little a little and then he
 - 35000 iterations: know what the man was going to be a little on the planet he said that he said and i think i think it was a very good time and the universe is a lot of a bit of the
 - 100000 iterations: said the old man who was a kind of paper of the universe he said i think i was a kind of thing to do it said ford with a shrug and then a bit of the universe i dont
 - 220000 iterations: at the same in the universe he said and i think i was a lot of this and i was going to do it said arthur i dont know said arthur i was a sort of thing that you were
 
Additional training iterations may yield additional improvements in grammar and semantics, but training was stopped for the sake of time.