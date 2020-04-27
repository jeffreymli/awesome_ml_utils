import pandas as pd
from scipy import stats
import sys
sys.path.insert(1, '../src/')
from pre_processing import *

def test_missing_values_table():
	X = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))

	X['A'].loc[10:] = np.nan ## 990
	X['B'].loc[500:] = np.nan ## 500
	X['C'].loc[400:] = np.nan ## 600
	X['D'].loc[900:] = np.nan ## 100

	mv = missing_values_table(X)
	
	mv[mv.index == 'A']['Missing Values']

	assert mv[mv.index == 'A']['Missing Values'][0] == 990
	assert mv[mv.index == 'B']['Missing Values'][0] == 500
	assert mv[mv.index == 'C']['Missing Values'][0] == 600
	assert mv[mv.index == 'D']['Missing Values'][0] == 100

def test_outlier_report():

	X = pd.DataFrame(np.random.randn(10000, 4), columns=list('ABCD'))
	outliers = [99999,99999,99999,99999]

	o = OutlierImputation(X, cont_cols= 'C',std_threshold = 3)
	o.fit()
	report = o.get_outlier_report()

def test_outlier_transformations():

	X = pd.DataFrame(np.random.randn(10000, 4), columns=list('ABCD'))

	### Winsor
	o = OutlierImputation(X, cont_cols= 'C',std_threshold = 2)
	o.fit()

	o.get_outlier_report()
	data = o.transform(impute_type = 'winsor', upper_bound_perc = 97.5, lower_bound_perc = 2.5)

	upper_bound = np.percentile(X['C'], 97.5)
	lower_bound = np.percentile(X['C'], 2.5)

	upper_bound_counts = X[X['C'] > upper_bound]['C']
	lower_bound_counts = X[X['C'] < lower_bound]['C']

	c = data[data['C'] > upper_bound]['C_out_imp']
	
	## Check winsor counts
	assert len(data[data['C'] > upper_bound]['C_out_imp']) == len(upper_bound_counts)
	assert len(data[data['C'] < lower_bound]['C_out_imp']) == len(lower_bound_counts)

	## Check winsor values
	assert round(np.mean(data[data['C'] < lower_bound]['C_out_imp']),3) == round(lower_bound,3)
	assert round(np.mean(data[data['C'] > upper_bound]['C_out_imp']),3) == round(upper_bound,3)
	
	## Check winsor imputed value
	assert max(data['C_out_imp']) == upper_bound
	assert min(data['C_out_imp']) == lower_bound

	### Check drop counts
	data = o.transform(impute_type = 'drop', upper_bound_perc = 97.5, lower_bound_perc = 2.5)
	assert len(upper_bound_counts) + len(lower_bound_counts) == len(X) - len(data)

	### Check multiple drop columns
	o = OutlierImputation(X, cont_cols= ['C','A','B'] ,std_threshold = 2)
	o.fit()
	data = o.transform(impute_type = 'drop', upper_bound_perc = 97.5, lower_bound_perc = 2.5)

	### Check custom function
	o = OutlierImputation(X, cont_cols= ['C','A','B'] ,std_threshold = 2)
	o.fit()
	data = o.transform(impute_type = 'custom', custom_value = 99999999, upper_bound_perc = 97.5, lower_bound_perc = 2.5)

	print(np.mean(data[data['C'] < lower_bound]['C_out_imp']))
	print(np.mean(data[data['C'] > upper_bound]['C_out_imp']) )
	assert np.mean(data[data['C'] < lower_bound]['C_out_imp']) == 99999999
	assert np.mean(data[data['C'] > upper_bound]['C_out_imp']) == 99999999


if __name__=='__main__':
	# test_outlier_report()
	test_outlier_transformations()
	

