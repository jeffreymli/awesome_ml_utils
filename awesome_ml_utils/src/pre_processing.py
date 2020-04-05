import pandas as pd
from scipy import stats
import numpy as np

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
           "There are " + str(mis_val_table_ren_columns.shape[0]) + 
           " columns that have missing values.")

    return mis_val_table_ren_columns


class OutlierImputation:
    def __init__(self, df, cont_cols, std_threshold):
        self.df = df
        self.std_threshold = std_threshold
        self.outliers = None
        
        if isinstance(cont_cols, str):
            self.cont_cols = [cont_cols]
        else:
            self.cont_cols = cont_cols

    def _get_outliers(self):
        if isinstance(self.cont_cols, str):
            self.cont_cols = [self.cont_cols]

        assert set(self.cont_cols).issubset(self.df.columns), 'Specified columns do not exist in dataset'

        self.cont_data = self.df[self.cont_cols].copy()
        self.z = pd.DataFrame(np.abs(stats.zscore(self.cont_data)))
        self.z.columns = self.cont_data.columns
        self.outliers = self.z[self.z[self.cont_cols] > self.std_threshold]


    def fit(self):
        self._get_outliers()

    def _calculate_transform(self, impute_type, upper_bound, lower_bound, custom_function = None):

        if upper_bound is None:
            ## Three standard deviations
            upper_bound = 99.85

        if lower_bound is None:
            lower_pound = 0.15
        
        for col in self.cont_cols:

            lower = np.percentile(self.df[col], lower_bound)
            upper = np.percentile(self.df[col], upper_bound)

            col_name = str(col) + '_out_imp'

            logging.info("Column name {0} is now called {1} with imputation".format(str(col),col_name))

            if impute_type = 'winsor':
                self.df.loc[:,col_name] = np.where(self.df[col] > upper, upper, self.df[col])
                self.df.loc[:,col_name] = np.where(self.df[col] > lower, lower, self.df[col])

            if impute_type = 'drop':

                self.df.loc[:,col] = np.where(self.df[col] > upper, 'drop_999999', self.df[col])
                self.df.loc[:,col] = np.where(self.df[col] > lower, 'drop_999999', self.df[col])
                self.df = self.df[self.df[col] != 'drop_999999'].copy()

            if impute_type = 'custom':

                self.df.loc[:,col_name] = np.where(self.df[col] > upper, custom_function(self.df[col]), self.df[col])
                self.df.loc[:,col_name] = np.where(self.df[col] > lower, custom_function(self.df[col]), self.df[col])
                

    def transform(self, impute_type, custom_function = None, upper_bound_perc = None, lower_bound_perc = None, **kwargs):
        
        if impute_type = 'winsor' or impute_type is None:
            print("Applying Winsor transformation.......")
            if upper_bound_perc is None:
                print("No upper bound percentile specified. Using default 3 standard deviations")
            if lower_bound_perc is None:
                print("No lower bound percentile specified. Using default 3 standard deviations")
            self._calculate_transform(impute_type, upper_bound_perc, lower_bound_perc)

        if impute_type = 'drop':
            print("Dropping all outliers. Be careful with this method......")
            if upper_bound_perc is None:
                print("No upper bound percentile specified. Using default 3 standard deviations")
            if lower_bound_perc is None:
                print("No lower bound percentile specified. Using default 3 standard deviations")
            self._calculate_transform(impute_type, upper_bound_perc, lower_bound_perc)

        if impute_type = 'custom':
            print("Using a custom function")
            if upper_bound_perc is None:
                print("No upper bound percentile specified. Using default 3 standard deviations")
            if lower_bound_perc is None:
                print("No lower bound percentile specified. Using default 3 standard deviations")
            self._calculate_transform(impute_type, upper_bound_perc, lower_bound_perc, **kwargs)


    def get_outlier_report(self):

        report = pd.DataFrame(self.outliers.count())
        report.loc[:,'outliers_pct'] = self.outliers.count()/self.df.count()
        report['mean'] = self.cont_data[self.outliers.notnull()].mean()
        report['min'] = self.cont_data[self.outliers.notnull()].min()
        report['max'] = self.cont_data[self.outliers.notnull()].max()
        report.columns = ['outlier_count','outliers_pct','outlier_mean','outlier_min','outlier_max']

        return report
        

            
