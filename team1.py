print("Team1")
import numpy as np

# indexing and slicing

a = np.array([10,20,30,40,50,60])

# indexing
print(a[0])
print(a[1])
print(a[-1])

# slicing


print(a[0:6:1])
print(a[-1:-7:-1])

# indexing + slicing

b= np.array([
    [1, 2, 3],
    [4, 5, 6],
    [8, 7, 9]
    ])

# print("b[0,3: ]",b[0:2,:])

# boolean marking

mask = (a>20) & (a<50)
c = a[mask]
print("mask: ",mask)
print("c",c)
