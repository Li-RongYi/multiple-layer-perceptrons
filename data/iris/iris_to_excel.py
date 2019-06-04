import pandas as pd
data_file = 'iris.data'

data=[]
file = open(data_file,'r')
for i in file.readlines()[:-1]:
    i=i.strip().split(',')
    #print(i)
    d=list(map(float,i[:-1]))
    c=i[-1]
    if c=='Iris-setosa':
        d.append(1)
    elif c=='Iris-versicolor':
        d.append(2)
    elif c=='Iris-virginica':
        d.append(3)
    #print(d)
    data.append(d)

file.close()

#columns = ['Sepal length','Sepal width','Petal length','Petal length','Class']
#df = pd.DataFrame(data=data,columns=columns)
df = pd.DataFrame(data=data,columns=None)
df.to_excel('iris.xlsx',index=False)
