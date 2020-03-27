import pandas as pd
from scipy import stats

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
        self.cont_cols = cont_cols
        self.std_threshold = std_threshold
        self.outliers = None

    def _get_outliers(self):
        if isinstance(self.cont_cols, str):
            self.cont_cols = [self.cont_cols]

        assert set(self.cont_cols).issubset(self.df.columns), 'Specified columns do not exist in dataset'

        cont_data = self.df[self.cont_cols].copy()
        z = pd.DataFrame(np.abs(stats.zscore(cont_data)))
        z.columns = cont_data.columns
        self.outliers = z[z[cont_cols] > std_threshold]

    def get_outlier_report(self):

        self._get_outliers()
        report = pd.DataFrame(self.outliers.count())
        report.loc[:,'outliers_pct'] = outliers.count()/df.count()
        report['mean'] = cont_data[outliers.notnull()].mean()
        report['min'] = cont_data[outliers.notnull()].min()
        report['max'] = cont_data[outliers.notnull()].max()
        report.columns = ['outlier_count','outliers_pct','outlier_mean','outlier_min','outlier_max']

        return report

    def fit(self):
        self._get_outliers()
        
    def transform(self, function, drop = False):

        for col in self.cont_cols:
            val = function(self.df[col])
            

        pass 