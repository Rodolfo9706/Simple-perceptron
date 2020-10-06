# Rodolfo Isaac Verdín Monzón
# 05/10/20
# Perceptron and,or,nor,nand


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time


def escalon(x_1):
  return np.where(x_1 >=0, 1, 0)


x = np.array([[0,0], [0,1],[1,0],[1,1]])
t = np.array([[1],[0],[0],[0]])

# Compuertas sustituir target por t
com_or = np.array([[0],[1],[1],[1]])
com_nor = np.array([[0],[1],[1],[1]])
com_and = np.array([[0],[1],[1],[1]])
com_nand = np.array([[0],[1],[1],[1]])

target = com_nor

n = len(t)
costo = []

#taza de aprendizaje
alpha = 0.5
#pesos aleatorios
w_i = np.random.rand(3,1)
epochs = 40

tic = time.time()
#añadiendo las columnas del bias  a x
bias = -np.ones((x.shape[0], 1))
x_1 = np.concatenate([x,bias], axis = 1)


for i in range(epochs):
  a = np.dot(x_1,w_i) 
  # print(a)
  yp = escalon(a)


  #regla del perceptron
  dw = ((t - yp).T.dot(x_1)).mean(axis=0)
  toc = time.time()
  wn = w_i.T + (alpha*dw)
  w_i = wn.T
  epochs = i
  costo.append(np.mean((t-yp)**2,axis=0))

  if costo[i] == 0:
    break

#grafica epocas vs costo 
print(epoch)
plt.plot(range(epochs+1), costo)
plt.show()

#Boundaries 
b, w1, w2 = w_i
ejex = -b / w1
ejey = -b / w2

d = ejey
c = -ejey / ejex

line_x_coords = np.array([0, ejex])
line_y_coords = c * line_x_coords + d

plt.plot(line_x_coords, line_y_coords)
plt.scatter(*x_1[:, 1:].T, c=t, s=75)
plt.show()

  
# Target vs Predicted
print('Perceptron rule')
print('target:    predicted:')
for j in range(len(yp)):
  res = str(t[j]) + '--------' + str(yp[j])
  print(res)

print('')
print('required time:{:.5f} s'.format(toc - tic))
print('required epochs: {}'.format(epoch))
