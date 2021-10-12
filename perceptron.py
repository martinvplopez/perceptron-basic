# Perceptron para clasificar distintas especies de flores en el dataset Iris. En este ejemplo sólo se clasificarán la setosa de la virginica teniendo en cuenta la longitud sepal y la del pétalo.
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
print("Perceptron")

class Perceptron(object):
    """
        Parametros:
        eta : float--> taza de aprendizaje (0.0-1.0)
        n_iter : int--> iteraciones en el conjunto de entrenamiento
        random_state : int--> generador aleatorio de pesos

        Atributos:
        w_ : 1d-array--> Pesos después del aprendizaje
        errors_ : list
        """
    def __init__(self, eta=0.01,n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        """Aprendizaje del modelo.
        Parametros:
        X : {array-like}, shape = [n_examples, n_features]--> vector de entrenamiento con número de muestras y de características
        y : array-like, shape = [n_examples]--> valores target
        Devuele un objeto
        """
        rgen= np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0, scale=0.01, size=1+x.shape[1]) #Pesos en base al número de características + el peso bias w0
        self.errors_=[]
        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(x,y):
                update=self.eta*(target-self.predict(xi)) # Se hace la actualización de los pesos(taza aprendizaje*(valor real-predicho))
                self.w_[1:] += update*xi # Se actualizan todos los pesos menos el bias.
                self.w_[0] += update  # x0 es 1.
                errors += int(update != 0.0) # Solo se añade el error si no es nulo
            self.errors_.append(errors)
            print(self.w_)
    def net_input(self,x):
        return np.dot(x,self.w_[1:]) + self.w_[0] # Realiza la suma pesada (producto escalar wx + w0)
    def predict(self,x):
        return np.where(self.net_input(x)>= 0.0, 1, -1)
        # Realiza la función de activacion. Se valora si la suma pesada es mayor o igual que 0, devuelve 1 y -1 en caso contrario.



# Importando dataset Iris con pandas
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None, encoding='utf-8')
# Extrearemos solamente dos etiquetas (flores setosa y versicolor), son las 100 primeras muestras.
# Sólo trabajaremos con la  las características longitud sepal y longitudo del pétalo (1era y 3era columna)
y= df.iloc[0:100,4].values
# La clasificación consistirá en setosa -1 y versicolor 1
y=np.where(y=='Iris-setosa', -1,1)
x= df.iloc[0:100,[0, 2]].values # Coger sólo la primera y tercera columna

# Iniciando Perceptron
ppn= Perceptron(eta=0.1,n_iter=10)
ppn.fit(x,y)



# Visualizacion con matplotlib

plt.scatter(x[:50,0], x[:50,1], color='red', marker='o', label='setosa (-1)') # Muestras de setosa con las dos columnas
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker='x', label='versicolor (1)') # Muestras de versicolor con las dos columnas
plt.xlabel('longitud sepal (cm)')
plt.ylabel('longitud de pétalo (cm)')
plt.legend(loc='upper left')
# Visualizacion de la clasificación
sepal_min, sepal_max = x[:, 0].min() - 1, x[:, 0].max() + 1 # Coger valores maximos y minimos de la longitud sepal
petalo_min, petalo_max = x[:, 1].min() - 1, x[:, 1].max() + 1  # Coger valores maximos y minimos de la longitud pétalo
# Se crea una matriz de puntos para ambas propiedades
valSepal, valPetalo = np.meshgrid(np.arange(sepal_min, sepal_max, 0.02),
                                  np.arange(petalo_min, petalo_max, 0.02))
# Se clasifican los valores (1 ó -1)
Z = ppn.predict(np.array([valSepal.ravel(), valPetalo.ravel()]).T)
Z = Z.reshape(valSepal.shape)

plt.contourf(valSepal, valPetalo, Z, alpha=0.3)
plt.xlim(valSepal.min(), valSepal.max())
plt.ylim(valPetalo.min(), valPetalo.max())
#plt.show()
plt.savefig('classificationPerceptron.png')
plt.close()


# Visualización de los errores
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Errors')
#plt.show()
plt.savefig('errorsPerceptron.png')



