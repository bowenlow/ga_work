from empath import Empath
import pickle
import pandas as pd
import numpy as np
import multiprocessing
# import multiprocessing_import_worker
import time
from os import getpid
from itertools import product



def gen_empath_worker_func(doc,num):
    lex = Empath()
    lex_results = pd.DataFrame(columns=lex.cats)
    lex_results = lex_results.append(pd.Series([np.nan]), ignore_index=True)
    results = lex.analyze(doc)
    for k in results.keys():
        lex_results[k] = results[k]
    # print("Task:" + str(num))
    return lex_results


if __name__ == '__main__':

	unlabelled_corpus = pickle.load(open('./input/unlabelled_corpus_clean.p','rb'))
	true_corpus = pickle.load(open('./input/true_corpus_clean.p','rb'))
	fake_corpus = pickle.load(open('./input/fake_corpus_clean.p','rb'))

	set_size = 100

	hasNext = True
	ctr = 0
	curr_corpus = true_corpus
	curr_fn = './input/true_empmp_vec'
	while hasNext:
		start = ctr * set_size
		end = min(start + set_size, len(curr_corpus))
		if (start+set_size >= len(curr_corpus)):
			hasNext = False

		st = time.time()
		pool = multiprocessing.Pool(processes = 8)
		results = pool.starmap(gen_empath_worker_func, zip(curr_corpus[start:end],
			range(len(curr_corpus[start:end]))))

		results_df = results[0]
		for a in range(1,len(results)):
			results_df = results_df.append(results[a], ignore_index=True)
		results_df.drop(columns=[0],inplace=True)
		et = time.time()
		print(et-st)

		pickle.dump(results_df, open(curr_fn+str(ctr)+'.p','wb'))
		print(results_df.head(3))
		pool.close()
		pool.join()
		ctr+=1

