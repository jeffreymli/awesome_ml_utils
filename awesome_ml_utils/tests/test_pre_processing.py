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

	missing_values_table = missing_values_table(X)

	assert c['Missing Values'].iloc[0] == 990
	assert c['Missing Values'].iloc[0] == 500
	assert c['Missing Values'].iloc[0] == 600
	assert c['Missing Values'].iloc[0] == 100

def test_detect_outliers():

	X = pd.DataFrame(np.random.randn(10000, 4), columns=list('ABCD'))
	outliers = [99999,99999,99999,99999]

	o = OutlierImputation(X, cont_cols= 'C',std_threshold = 3)
	report = o.get_outlier_report()
	print(report)








if __name__=='__main__':
	test_detect_outliers()
	

