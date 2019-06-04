import pandas as pd

data_file = 'glass.data'

data = []
file = open(data_file, 'r')
for i in file.readlines():
    i = i.strip().split(',')
    print(i)
    d = list(map(float, i[1:-1]))
    d.append(int(i[-1]))
    data.append(d)

file.close()

df = pd.DataFrame(data=data, columns=None)
df.to_excel('glass.xlsx', index=False)
