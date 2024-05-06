# import library
import pandas as pd
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("linear-regression-dataset.csv", sep = ";") # araları noktalı virgülle ayır

#plot plot
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# sklearn library
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

# pandası numpy arraye çeviriyoruz
x = df.deneyim.values.reshape(-1,1) #reshape etmemizin amacı boyutu (14,) olarak yazan x.shapei (14, 1) olacak şekile çevirmek..
y = df.maas.values.reshape(-1,1)    #çünkü sklearn kütüphanesi bu şekiklde tanımlanıyor.

linear_reg.fit(x,y) # line'ımızı fit ettik

y_head = linear_reg.predict(x) # predict ettiğim maaşım

plt.plot(x, y_head, color = "red")

from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y, y_head))
