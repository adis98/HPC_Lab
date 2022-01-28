import numpy as np
m = int(input('enter the number of inputs'))
y = np.zeros((m,1))
x = np.zeros((m,1))
#x = np.array([[51.3],[49.9],[50],[49.2],[48.5],[47.8],[47.3],[45.1],[46.3],[42.1],[44.2],[43.5],[42.3],[40.2],[31.8],[34]])
#y=np.array([[102.5],[104.5],[100.4],[95.9],[87],[95],[88.6],[89.2],[78.9],[84.6],[81.7],[72.2],[65.1],[68.1],[67.3],[52.5]])

print('enter x')
for i in range(m):
    x[i] = float(input(':'))
print('enter y')
for i in range(m):
    y[i] = float(input(':'))
#print(y)
#assume b0 +b1X = y_hat
t1 = np.mean(x*y)
t2 = np.mean(x) * np.mean(y)
t3 = np.mean(x*x)-(np.mean(x)**2)
b1 = (t1-t2)/t3
b0 = np.mean(y)-b1*(np.mean(x))
print(b0,b1)

#y_hat = b0 + b1*x

#tss = np.sum(y_hat*y_hat)
#residuals = y-y_hat
#sse = np.sum(residuals*residuals)
#msse = sse/(m-2) #in this case since it only depends on Temperature
"""print(sse,msse)

y_mean = np.mean(y)
ctss = (np.sum((y-y_mean)*(y-y_mean)))

regressor_ss = ctss-sse

f_stat = regressor_ss/msse
print(f_stat)

mortality_index = regressor_ss/ctss
print(mortality_index)"""
