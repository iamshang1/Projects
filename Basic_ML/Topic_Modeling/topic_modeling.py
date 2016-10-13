import tarfile
from gensim import corpora, models, matutils
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#load posts
print "loading posts"
posts_gzip = tarfile.open('20_newsgroups.tar.gz', 'r:gz')
posts = []
for f in posts_gzip.getnames():
    try:
        posts.append(posts_gzip.extractfile(f).read())
    except:
        pass
posts_gzip.close()

#vectorize posts
print "vectorizing posts"
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
vectorizer = StemmedTfidfVectorizer(min_df=2, stop_words='english', ngram_range=(1, 2), decode_error='ignore')
vector = vectorizer.fit_transform(posts)

#create gensim corpus and dictionary
print "creating corpus and dictionary"
corpus = matutils.Sparse2Corpus(vector, documents_columns=False)
dictionary = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())

#build lda model
print "building latent dirichlet allocation model"
lda = models.ldamodel.LdaModel(corpus, num_topics=20, id2word=dictionary, eta=0.01, alpha=0.25)
topics = matutils.corpus2dense(lda[corpus], num_terms=lda.num_topics)
weight = topics.sum(1)
max_topics = weight.argsort()[-3:][::-1]

#display word cloud of 3 most common topics
for i in range(len(max_topics)):
    words = lda.show_topic(max_topics[i], 50)
    words = [(str(w[0]) + " ")*int(w[1]*10000) for w in words]
    words = " ".join(words)
    wordcloud = WordCloud(max_font_size=50, relative_scaling=.5).generate(words)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("wordcloud%s.png" % str(i+1))
    plt.show()