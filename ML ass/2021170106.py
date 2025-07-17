
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn import metrics
def calc(x,y,m,c,n,l,epochs,s):
    for i in range(epochs):
        y_pred = m * x + c
        dm = (-2 / n) * sum((y - y_pred) * x)
        dc = (2 / n) * sum(y - y_pred)
        m = m - l * dm
        c = c - l * dc
    pred = m * x + c

    plt.scatter(x, y)
    plt.xlabel(s)
    plt.ylabel("Performance Index")
    plt.plot(x, pred, 'red', linewidth=4)
    plt.show()
    print("mean squared error of",s," :", metrics.mean_squared_error(y, pred))

data=pd.read_csv("assignment1dataset.csv")
print(data.describe())
x=data["Hours Studied"]
y=data["Performance Index"]
plt.scatter(x, y)
plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.show()
s="Hours Studied"
l=0.001
m=-25
c=45
epochs=1000
n=int(len(x))
calc(x,y,m,c,n,l,epochs,s)
x=data["Sleep Hours"]
y=data["Performance Index"]
plt.scatter(x, y)
plt.xlabel("Sleep Hours")
plt.ylabel("Performance Index")
plt.show()
s="Sleep Hours"
l=0.0001
m=-20
c=54
epochs=1000
n=int(len(x))
calc(x,y,m,c,n,l,epochs,s)

x=data["Sample Question Papers Practiced"]
y=data["Performance Index"]
plt.scatter(x, y)
plt.xlabel("Sample Question Papers Practiced")
plt.ylabel("Performance Index")
plt.show()
s="Sample Question Papers Practiced"
l=0.0001
m=-25
c=57
epochs=1000
n=int(len(x))
calc(x,y,m,c,n,l,epochs,s)
x=data["Previous Scores"]
y=data["Performance Index"]
plt.scatter(x, y)
plt.xlabel("Previous Scores")
plt.ylabel("Performance Index")
plt.show()
s="Previous Scores"
l=0.0001
m=-30
c=-15
epochs=800
n=int(len(x))
calc(x,y,m,c,n,l,epochs,s)
data["Hours Studied + Previous Scores"]=data["Hours Studied"]+data["Previous Scores"]
x=data["Hours Studied + Previous Scores"]
y=data["Performance Index"]
plt.scatter(x, y)
plt.xlabel("Hours Studied + Previous Scores")
plt.ylabel("Performance Index")
plt.show()
s="Hours Studied + Previous Scores"
l=0.0001
m=-20
c=-23
epochs=1800
n=int(len(x))
calc(x,y,m,c,n,l,epochs,s)
