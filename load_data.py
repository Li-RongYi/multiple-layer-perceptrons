import pandas as pd
import numpy as np
from sklearn import preprocessing

class Data():
    def load(self,data_name='iris',data_process='None'):
        if data_name=='iris':
            file_name = 'data/iris/iris.xlsx'
        elif data_name=='wine':
            file_name = 'data/wine/wine.xlsx'
        elif data_name=='glass':
            file_name = 'data/glass/glass.xlsx'
        try:
            df = pd.read_excel(file_name)
        except Exception as e:
            return None,e

        data = np.asarray(df[df.columns[:-1]])
        #print(data)
        label = np.asarray(df[df.columns[-1]]).ravel()
        #print(label)

        if data_process=='Normalization':
            data=preprocessing.normalize(data)
        elif data_process=='Standardization':
            data=preprocessing.scale(data)

        return data,label
