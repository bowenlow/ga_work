import flask
from forms import LoginForm, ReviewForm
from config import Config
from flask import url_for, redirect, flash
from logging.config import dictConfig

import pandas as pd
import numpy as np
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import zipfile
import urllib.request
from io import BytesIO
import gensim
from gensim import corpora, models, similarities

np.random.seed(0)

#CONFIG
app = flask.Flask(__name__) # init Flask 
app.config.from_object(Config)

def gen_vader_vec(sentence):
	compound = []
	neg = []
	neu = []
	pos = []
	sent = gl_sia.polarity_scores(sentence)
	compound.append(sent['compound'])
	neg.append(sent['neg'])
	neu.append(sent['neu'])
	pos.append(sent['pos'])
	vader_sent = pd.DataFrame({'compound':compound, 'neg':neg, 'neu':neu, 'pos':pos})
	return vader_sent

def gen_empath_vec(sentence):
	lexicon_results = pd.DataFrame(columns=gl_lexicon.cats)
	lexicon_results = lexicon_results.append(pd.Series([np.nan]), ignore_index=True)
	results = gl_lexicon.analyze(sentence)
	for k in results.keys():
		lexicon_results[k].iloc[0] = results[k]
	# Drop extra column generated
	lexicon_results.drop(columns=[0], inplace=True)
	return lexicon_results


def gen_lda_vec(bow):
	num_topics = gl_lda_model.num_topics
	topic_dists = np.zeros([1, num_topics])
	dists = gl_lda_model.get_document_topics(bow)
	indices = list(dict(dists).keys())        
	vals = list(dict(dists).values()) 
	topic_dists[0,indices] = vals
	topic_dists = pd.DataFrame(topic_dists, columns = ['topic'+str(a) for a in range(num_topics)])
	return topic_dists

#Routes
@app.route('/review', methods =['GET','POST'])
def show_form():
	form = ReviewForm()
	if form.validate_on_submit():
		# preprocess text
		raw = list(gensim.utils.tokenize(form.reviewtext.data))

		cleaned = gl_bigrams[raw]
		cleaned = [c for c in cleaned if c not in gl_all_stopset]
		cleaned_doc = gensim.models.doc2vec.TaggedDocument(cleaned,[0])

		# gen doc2vec word embeddings
		d2v_vec = pd.DataFrame(gl_doc2vec_model.infer_vector(cleaned_doc.words))
		app.logger.info(d2v_vec)

		# gen VADER vec
		vader_vec = gen_vader_vec(' '.join(raw))
		app.logger.info(vader_vec)
		
		# gen Empath vec
		emp_vec = gen_empath_vec(' '.join(raw))
		app.logger.info(emp_vec)
		# gen LDA
		cleaned_bow = gl_corpus_dict.doc2bow(cleaned)
		lda_vec = gen_lda_vec(cleaned_bow)
		app.logger.info(lda_vec)

		test_vec = pd.concat([d2v_vec.T, vader_vec, lda_vec, emp_vec], axis=1)

		probs = gl_xg_clf.predict_proba(test_vec)
		probs2 = gl_xg_clf.predict(test_vec)
		app.logger.info(raw)
		app.logger.info(cleaned_doc.words)
		app.logger.info(probs)
		flash(str(probs))
		return redirect(url_for('show_form'))
	return flask.render_template('review.html', form=form)

gl_bigrams = None
gl_sia = None
gl_lexicon = None
gl_lda_model = None
gl_corpus_dict = None
gl_doc2vec_model = None
gl_all_stopset = None
gl_xg_clf = None

#Main Sentinel
if __name__ == '__main__':
	#load bigrams
	gl_bigrams = pickle.load(open('plain_bigram.p','rb'))
	# load VADER 
	gl_sia = SentimentIntensityAnalyzer()
	# load Empath
	gl_lexicon = Empath()

	# load LDA model
	lda_url = "https://www.dropbox.com/s/l4tmm8krbmz2fl0/lda_model.zip?dl=1"
	app.logger.info('Opening LDA Url')
	response = urllib.request.urlopen(lda_url)
	app.logger.info('Reading LDA response')
	with zipfile.ZipFile(BytesIO(response.read())) as myzip:
		app.logger.info('Unzipping LDA')
		with myzip.open('lda_model.p') as myfile:
			gl_lda_model = pickle.load(myfile)

	# gl_lda_model = pickle.load(open('lda_model.p','rb'))

	# load corpus dict
	gl_corpus_dict = pickle.load(open('corpus_dict.p','rb'))

	# load doc2vec model
	doc2vec_url = "https://www.dropbox.com/s/s1zoaoeyguwcpqp/doc2vec_model.zip?dl=1"
	app.logger.info('Opening doc2vec Url')
	response = urllib.request.urlopen(doc2vec_url)
	app.logger.info('Reading D2V response')
	with zipfile.ZipFile(BytesIO(response.read())) as myzip:
		app.logger.info('Unzipping D2V')
		with myzip.open('doc2vec_model.p') as myfile:
			gl_doc2vec_model = pickle.load(myfile)

	# gl_doc2vec_model = pickle.load(open('doc2vec_model.p','rb'))

	# load stopwords
	stop_words = pd.read_csv('stopwords.csv',names=['stop'])
	new_stop = stop_words.stop.map(lambda x: str.capitalize(x))
	all_stop_set = set(stop_words.stop.append(new_stop,ignore_index=True))

	#generate combinations of all_stop_words, bigrams
	#generate bigram
	bigram_stops=[]
	for a in all_stop_set:
	    for b in all_stop_set:
	        bigram_stop = a+"_"+b
	        bigram_stops.append(bigram_stop)
	gl_all_stopset = all_stop_set.union(bigram_stops)

	# load classifier
	gl_xg_clf = pickle.load(open('best_model.p','rb'))
	app.run()