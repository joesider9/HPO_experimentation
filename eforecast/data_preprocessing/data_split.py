import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from eforecast.dataset_creation.files_manager import FilesManager
from eforecast.common_utils.dataset_utils import split_data_val


class Splitter:
    def __init__(self, static_data, is_online=False, train=False):
        self.static_data = static_data
        self.is_online = is_online
        self.train = train
        self.is_fuzzy = self.static_data['is_Fuzzy']
        self.is_Global = self.static_data['is_Global']
        self.val_test_ratio = self.static_data['val_test_ratio']
        self.file_manager = FilesManager(static_data, is_online=is_online, train=train)

    def split_cluster_data(self, activations):
        mask_train_temp, mask_test = split_data_val(activations, test_size=self.val_test_ratio, random_state=42,
                                                    thes_act=self.static_data['clustering']['thres_act'],
                                                    continuous=True)
        mask_train, mask_val = split_data_val(activations, test_size=self.val_test_ratio, random_state=42,
                                              thes_act=self.static_data['clustering']['thres_act'],
                                              continuous=False,
                                              dates_freeze=mask_test)
        if not isinstance(mask_train, pd.DatetimeIndex):
            mask_train = pd.DatetimeIndex(mask_train)
        if not isinstance(mask_val, pd.DatetimeIndex):
            mask_val = pd.DatetimeIndex(mask_val)
        if not isinstance(mask_test, pd.DatetimeIndex):
            mask_test = pd.DatetimeIndex(mask_test)
        return [mask_train, mask_val, mask_test]

    def split(self, refit=False):
        if refit:
            self.file_manager.remove_cv_data_files()
        cv_mask = self.file_manager.check_if_exists_cv_data()
        if cv_mask is None:
            data_row = self.file_manager.check_if_exists_row_data()
            if data_row is None:
                raise ImportError('Cannot find data row to split. Check if data are exists')
            y = self.file_manager.check_if_exists_target()
            if y is None:
                raise ImportError('Cannot find target data to stratify for split. Check if data are exists')
            split_test = int(data_row.shape[0] * (1 - self.val_test_ratio))
            mask_test = data_row.index[split_test:]
            mask_train_temp = data_row.index[:split_test]
            y_temp = y.iloc[:split_test]
            binarizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
            y_tr = binarizer.fit_transform(y_temp.values)
            y_tr = pd.DataFrame(y_tr, index=y_temp.index, columns=y_temp.columns)
            mask_train, mask_val = train_test_split(mask_train_temp, test_size=self.val_test_ratio,
                                                    random_state=42,
                                                    stratify=y_tr)

            if not isinstance(mask_train, pd.DatetimeIndex):
                mask_train = pd.DatetimeIndex(mask_train)
            if not isinstance(mask_val, pd.DatetimeIndex):
                mask_val = pd.DatetimeIndex(mask_val)
            if not isinstance(mask_test, pd.DatetimeIndex):
                mask_test = pd.DatetimeIndex(mask_test)
            mask_train = mask_train.sort_values()
            mask_val = mask_val.sort_values()
            mask_test = mask_test.sort_values()
            self.file_manager.save_cv_data([mask_train, mask_val, mask_test])
