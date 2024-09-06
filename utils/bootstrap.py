import numpy as np

def bootstrap(dataframe, sample_size=None):
    if(sample_size == None):
        sample_size = len(dataframe)

    bootSample_i = (np.random.rand(sample_size) * len(dataframe)).astype(int)
    bootSample_i = np.array(bootSample_i)
    bootSample_dataframe = dataframe.iloc[bootSample_i]
    return bootSample_dataframe