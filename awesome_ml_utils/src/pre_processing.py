import pandas as pd
from scipy import stats
import numpy as np
import logging


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
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

        assert set(
            self.cont_cols).issubset(
            self.df.columns), 'Specified columns do not exist in dataset'

        self.cont_data = self.df[self.cont_cols].copy()
        self.z = pd.DataFrame(np.abs(stats.zscore(self.cont_data)))
        self.z.columns = self.cont_data.columns
        self.outliers = self.z[self.z[self.cont_cols] > self.std_threshold]

    def fit(self):
        self._get_outliers()

    def _calculate_transform(
            self,
            impute_type,
            upper_bound,
            lower_bound,
            custom_value=None):

        if upper_bound is None:
            # Three standard deviations
            upper_bound = 99.85

        if lower_bound is None:
            lower_bound = 0.15

        for col in self.cont_cols:

            original_df = self.df.copy()

            lower = np.percentile(original_df[col], lower_bound)
            upper = np.percentile(original_df[col], upper_bound)

            col_name = str(col) + '_out_imp'
            if impute_type == 'winsor':
                print(
                    "Column name {0} is now called {1} with imputation".format(
                        str(col), col_name))

                self.df.loc[:, col_name] = np.where(
                    self.df[col] > upper, upper, self.df[col])
                
                self.df.loc[:, col_name] = np.where(
                    self.df[col_name] < lower, lower,self.df[col_name])

            if impute_type == 'drop':

                total_length = len(self.df[col])
                self.df = self.df[self.df[col] <= upper].copy()
                dropped_upper_counts = total_length - len(self.df[col].values)

                print("Dropping {0} Upper Bound Values for column {1}".format(dropped_upper_counts, col))

                self.df = self.df[self.df[col] >= lower].copy()
                dropped_lower_counts = total_length - dropped_upper_counts - len(self.df[col])                
                
                print("Dropping {0} Lower Bound Values for column {1}".format(dropped_lower_counts, col))

                print(
                    "We have dropped a total of {0} values from column {1} ".format(
                        str(dropped_lower_counts + dropped_upper_counts), col_name))
                print('\n')
            if impute_type == 'custom':
                print(
                    "Column name {0} is now called {1} with imputation".format(
                        str(col), col_name))

                self.df.loc[:, col_name] = np.where(
                    self.df[col] > upper, custom_value, self.df[col])


                
                self.df.loc[:, col_name] = np.where(
                    self.df[col_name] < lower, custom_value,self.df[col_name])

    def transform(
            self,
            impute_type,
            custom_value=None,
            upper_bound_perc=None,
            lower_bound_perc=None,
            **kwargs):

        if impute_type == 'winsor' or impute_type is None:
            print("Applying Winsor transformation.......")
            if upper_bound_perc is None:
                print(
                    "No upper bound percentile specified. Using default 3 standard deviations")
            if lower_bound_perc is None:
                print(
                    "No lower bound percentile specified. Using default 3 standard deviations")
            self._calculate_transform(
                impute_type, upper_bound_perc, lower_bound_perc)

        if impute_type == 'drop':
            print("Dropping all outliers........")
            print('\n')

            if len(self.cont_cols) > 1:
                print("You specified more than one column.")
                print("Be careful with this method. We will drop all rows containing outliers starting with the first column.....")
                print("This means we may be dropping more values than we want")
                print('\n')

            if upper_bound_perc is None:
                print(
                    "No upper bound percentile specified. Using default 3 standard deviations")
            
            if lower_bound_perc is None:
                print(
                    "No lower bound percentile specified. Using default 3 standard deviations")
            self._calculate_transform(
                impute_type, upper_bound_perc, lower_bound_perc)

        if impute_type == 'custom':

            print("Using a custom function")
            if upper_bound_perc is None:
                print(
                    "No upper bound percentile specified. Using default 3 standard deviations")
            if lower_bound_perc is None:
                print(
                    "No lower bound percentile specified. Using default 3 standard deviations")
            self._calculate_transform(
                impute_type,
                upper_bound_perc,
                lower_bound_perc,
                custom_value,
                **kwargs)

        return self.df.copy()

    def get_outlier_report(self):

        report = pd.DataFrame(self.outliers.count())
        report.loc[:, 'outliers_pct'] = self.outliers.count() / self.df.count()
        report['mean'] = self.cont_data[self.outliers.notnull()].mean()
        report['min'] = self.cont_data[self.outliers.notnull()].min()
        report['max'] = self.cont_data[self.outliers.notnull()].max()
        report.columns = [
            'outlier_count',
            'outliers_pct',
            'outlier_mean',
            'outlier_min',
            'outlier_max']

        return report
