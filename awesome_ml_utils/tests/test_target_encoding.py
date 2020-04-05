import pandas as pd
import numpy as np
import sys
import time

sys.path.insert(1, '../src/')
from target_encoding import TargetEncoder

def test_column_equality():
    
	X = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))
	X['category'] = np.random.choice([-5, 5, 10], size=len(X))
	y = pd.Series(np.random.randn(1000))

	for i in range(2, 10):
		te = TargetEncoder(cols=['category'], n_folds = i)
		te.fit(X,y)
		encoded_data = te.transform(X,y)
		assert len(set(X['category']))*i == len(set(encoded_data['category'])), 'Encoded data does not have matching unique values'
		
	
def test_mean_vals_for_regular():
	X = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
	X['category'] = np.random.choice([-5, 5, 10], size=len(X))
	X.loc[:,'category_orig'] = X['category']
	y = pd.Series(np.random.randn(50))
	
	te = TargetEncoder(cols=['category'])
	te.fit(X,y)
	encoded_data = te.transform(X,y)

	X.loc[:,'target'] = y
	grouped_data = X.groupby(['category']).mean().reset_index()
	means_to_test = grouped_data[['category','target']]

	check_data = encoded_data.merge(means_to_test, 
		left_on='category_orig', 
		right_on ='category', how='left')
	# assert check_data['neighbourhood_group_x'] == check_data['number_of_reviews'], 'Mean values do not match up'
	te_vals = np.array(check_data['category_x'])
	original_vals = np.array(check_data['target'])
	
	assert np.array_equal(te_vals,original_vals) == True, ' Means values are not equal'


def test_fallback():
	X = pd.DataFrame(np.random.randint(0,10,size =(50,4)), columns=list('ABCD'))
	X['category'] = np.random.choice([-5, 5, 10], size=len(X))
	X.loc[:,'category_orig'] = X['category']
	
	y = pd.Series(np.random.randn(50))
	X['target'] = y

	y = np.where((X.category == 5), np.nan, X.target)
	te = TargetEncoder(cols=['category'])
	te.fit(X,y)
	encoded_data = te.transform(X,y)

	cat_5_val = np.mean(encoded_data[encoded_data['category_orig'] == 5]['category'])
	assert round(cat_5_val, 2) == round(te.fallback, 2), 'Fallback amount does not match'


def test_kfold():
	y = pd.DataFrame(np.random.randn(20, 1), columns=['target'])
	y = y.sample(frac=1, random_state = 123)

	zero_data = np.zeros(shape=(20,1))
	X = pd.DataFrame(zero_data)
	X.columns = ['category']

	X.loc[:5] = 'A'
	X.loc[5:10] = 'B'
	X.loc[10:15] = 'C'
	X.loc[15:] = 'D'

	X.loc[:,'category_orig'] = X['category']
	X = X.sample(frac=1, random_state = 123)	

	n_folds = 4
	te = TargetEncoder(cols=['category'], n_folds = n_folds)
	te.fit(X,y)
	results = te.transform(X,y)
	
	X.loc[:,'target'] = y
	num_fold_in_df = set(te._kfold_numbering(X,y,n_folds=4)['fold'])

	## Manually calculating target encoded fold values
	fold_1_a = np.mean([X.loc[0]['target'],X.loc[3]['target'],X.loc[1]['target'],X.loc[2]['target']])
	fold_2_b = np.mean([X.loc[5]['target'],X.loc[8]['target'],X.loc[9]['target'],X.loc[6]['target']])
	fold_3_c = np.mean([X.loc[14]['target'],X.loc[12]['target'],X.loc[13]['target']])
	fold_4_d = np.mean([X.loc[17]['target'],X.loc[19]['target'],X.loc[15]['target'],X.loc[16]['target']])

	assert len(num_fold_in_df) == n_folds, "Please double check the fold counts"
	assert results.loc[4]['category'] == fold_1_a
	assert results.loc[7]['category'] == fold_2_b
	assert results.loc[11]['category'] == fold_3_c
	assert results.loc[18]['category'] == fold_4_d



if __name__=='__main__':
	
	test_column_equality()
	# test_mean_vals_for_regular()
	# test_fallback()
	test_kfold()