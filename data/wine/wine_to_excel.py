import pandas as pd

data_file = 'wine.data'

data = []
file = open(data_file, 'r')
for i in file.readlines():
    i = i.strip().split(',')

    d = list(map(float, i[1:]))
    d.append(int(i[0]))
    data.append(d)

file.close()

df = pd.DataFrame(data=data, columns=None)
df.to_excel('wine.xlsx', index=False)
