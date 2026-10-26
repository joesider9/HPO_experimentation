import numpy as np
import pandas as pd


class LoadCompressor:
    def __init__(self, static_data, nwp_data, nwp_metadata):
        self.static_data = static_data
        self.horizon = self.static_data['horizon']
        self.use_data_before_and_after_target = self.static_data['use_data_before_and_after_target']
        self.type = self.static_data['type']
        self.nwp_metadata = nwp_metadata
        self.nwp_data = nwp_data

    def load_compressor(self, data):
        shape = data.shape
        data = data.reshape(-1, np.prod(shape[1:]))
        return np.mean(data, axis=1)

    def perform_load_compress(self, i, ax, nwp_data):
        if self.horizon == 0:
            ax_name = '_'.join(ax)
            data = self.load_compressor(nwp_data[:, :, :, :, i])
            data = pd.DataFrame(data, index=self.nwp_metadata['dates'], columns=[ax_name])
            nwp_compressed = data
        else:
            nwp_compressed = pd.DataFrame()
            for hor in range(self.horizon):
                ax_name = '_'.join(ax + [str(hor)])
                data = self.load_compressor(nwp_data[:, hor, :, :, i])
                data = pd.DataFrame(data, index=self.nwp_metadata['dates'], columns=[ax_name])
                nwp_compressed = pd.concat([nwp_compressed, data], axis=1)
        return nwp_compressed

    def load_compress(self):
        groups = self.nwp_metadata['groups']
        axis = self.nwp_metadata['axis']
        if len(groups) == 0:
            nwp_compressed = pd.DataFrame()
            nwp_compressed_all = pd.DataFrame()
            for i, ax in enumerate(axis):
                data = self.perform_load_compress(i, ax, self.nwp_data)
                nwp_compressed = pd.concat([nwp_compressed, data], axis=1)
            nwp_compressed_all = nwp_compressed
            nwp_compressed_distributed = np.mean(self.nwp_data, axis=(2, 3))
        else:
            nwp_compressed = dict()
            nwp_compressed_distributed = dict()
            for group in groups:
                group_name = '_'.join(group) if isinstance(group, tuple) else group
                nwp_compressed[group_name] = pd.DataFrame()
                for i, ax in enumerate(axis[group_name]):
                    data = self.perform_load_compress(i, ax, self.nwp_data[group_name])
                    nwp_compressed[group_name] = pd.concat([nwp_compressed[group_name], data], axis=1)
                nwp_compressed_distributed[group_name] = np.mean(self.nwp_data[group_name], axis=(2, 3))
            nwp_compressed_all = pd.DataFrame()
            for group_name, data in nwp_compressed.items():
                nwp_compressed_all = pd.concat([nwp_compressed_all, data], axis=1)
        return nwp_compressed_all, nwp_compressed, nwp_compressed_distributed
