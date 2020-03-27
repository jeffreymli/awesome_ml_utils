import pandas as pd


class TargetEncoder:
    def __init__(self, cols=None, n_folds=None):
        # default to encoding all categorical columns
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

        self.n_folds = n_folds
        self.kfold_means = dict()
        self.target_means = dict()
        self.fallback = None

    def _reg_fit(self, X, y):
        # Encode all categorical columns if not specified
        if self.cols is None:
            cols = []
            for i, col in enumerate(X.dtypes):
                if col == 'object':
                    cols.append(X.columns[i])
            self.cols = cols

        if self.n_folds is not None:
            self.target_means = dict()

        df = X.copy()
        df['target'] = y.astype('float')

        self.fallback = df['target'].mean()

        for cat_col in self.cols:
            target = df[[cat_col, 'target']].groupby(cat_col).mean().to_dict()['target']
            self.target_means[cat_col] = target
        return self

    def _reg_transform(self, X, y):
        X_transform = X.copy()
        cols = []
        ls_target_encodings = []

        # Likely a way we can impute all the eligible columns without looping
        # through each one
        for col, val in self.target_means.items():
            cols.append(col)
            ls_target_encodings.append(
                X_transform[col].map(val).fillna(
                    self.fallback))

        X_transform[cols] = pd.concat(
            ls_target_encodings,
            axis='columns',
            ignore_index=True)

        return X_transform

    def _kfold_numbering(self, X, y, n_folds):

        # Is there a way I can vectorize this for loop?
        df = X.copy()
        df['target'] = y.astype('float')
        parts = int(len(df) / n_folds)

        kfold_df = []
        for i in range(n_folds):
            if i == n_folds:
                break
            else:
                sliced_df = df[i * parts:(i + 1) * parts]
                sliced_df['fold'] = i
                kfold_df.append(sliced_df)
        kfold_df = pd.concat(kfold_df)
        return kfold_df

    def _kfold_fit(self, X, y):
        
        kfold_df = self._kfold_numbering(X, y, self.n_folds)
        for i in range(self.n_folds):
            # To Do: Figure out how to avoid re-computing these values in a few
            # lines of code.
            data_for_mean = kfold_df[kfold_df['fold'] != i]

            data_for_mean_X = data_for_mean.drop(['target'], axis=1)
            data_for_mean_y = data_for_mean['target']
            

            self._reg_fit(data_for_mean_X, data_for_mean_y)
            self.kfold_means[i] = self.target_means
            
        return self

    def _kfold_transform(self, X, y):
        kfold_df = self._kfold_numbering(X, y, self.n_folds)
        X_transform = []

        for i in self.kfold_means.keys():
            data_to_impute = kfold_df[kfold_df['fold'] == i]
            data_to_impute_X = data_to_impute.drop(['target'], axis=1)
            data_to_impute_y = data_to_impute['target']

            # To Do: See if I can map to fold numbers to improve speed
            self.target_means = self.kfold_means[i]
            fold_transform_X = self._reg_transform(
                data_to_impute_X, data_to_impute_y)

            X_transform.append(fold_transform_X)
        return pd.concat(X_transform)

    def fit(self, X, y):
        if self.n_folds is not None:
            assert self.n_folds < len(X)/self.n_folds, "Dataset not large enough to support specified number of folds"
            assert self.n_folds < len(X)/self.n_folds, "Dataset not large enough to support specified number of folds"
            self._kfold_fit(X, y)
        self._reg_fit(X, y)

    def transform(self, X, y):
        if self.n_folds is not None:
            return self._kfold_transform(X, y)
        return self._reg_transform(X, y)

    def fit_transform(self, X, y):
        if self.n_folds is not None:
            return self._kfold_fit(X, y, self.n_folds)._kfold_transform(X, y)
        return self.fit(X, y).transform(X, y)
