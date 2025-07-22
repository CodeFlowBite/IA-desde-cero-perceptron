import matplotlib.pyplot as plt
import random as rd

# --Clase neurona--
class perceptron:
    def __init__(self, x_input):
        self.w = [rd.uniform(-1, 1) for _ in range(x_input)]
        self.b = rd.uniform(-1, 1)
        self.x = []
    
    def forward(self, x):
        self.x = x
        return sum([x[_x] * self.w[_x] for _x in range(len(x))]) + self.b

    def update_parameters(self, costo, tz):
        self.w = [self.w[_x] - (costo * tz * self.x[_x]) for _x in range(len(self.w))]
        self.b = self.b - costo * tz

# --Modelo (Neurona) 
n1 = perceptron(1)

# --Crear datos para entrenar a la neurona--
#       --Convertir grados C° a F°--
print('-'*64)

data_x = [0, 10, 20, 30, 40, 50]                               # Una lista de grados C° // sera la entrada
data_y = [(_x * 1.8) + 32 for _x in data_x]                    # Una lista de grados F° // sera la salida

print(f"Datos de entrada cargados: {data_x}")
print(f"Datos de  salida cargados: {data_y}")
print('-'*64)

# --Normalizar datos

def Normalizar(data):
    maximo = max(data)
    return [data[z] / maximo for z in range(len(data))]

Data_x_norm = Normalizar(data_x)
Data_y_norm = Normalizar(data_y)

print(f"Datos de entrada normalizados: {Data_x_norm}")
print(f"Datos de  salida normalizados: {Data_y_norm}")
print('-'*64)

# --Crear funciones para el entrenamiento--

def Costo(y_pre, y_true):
    return y_pre - y_true

def perdida_cuadratica_media(costos):
    return (sum(costos) / len(costos))**2

# --Entrenamiento--

# --Declarar variables necesarias para el entrenamiento
epocas = 20   # Numero de veces que la neurna se entrenara con los datos de ejemplo
tz = 0.5      # La taza de aprendizaje le dice a la neurona cuanto debe que ajustar los parametros

error_epoca = [] # Lista que guardara el error para la visualizacion en la grafica
epoca_error = [] # Lista que guardara la epoca para la visualizacion en la grafica
pesos_error = [] # Lista que guardara el peso  para la visualizacion en la grafica
visualizar_graficas = False

epoca_revision = epocas # Graficar el progreso del modelo x veces

for epoch in range(epocas):
    prediccion_neurona = []
    Error_total = 0
    costo_d = []

    for d in range(len(Data_x_norm)): # Almacenamos los datos del costo sin actualizar al modelo (solo para visualizar en grafica del entrenamiento del modelo y la perdida total)
        y = n1.forward([Data_x_norm[d]])
        costo = Costo(y, Data_y_norm[d])
        costo_d.append(costo)
        prediccion_neurona.append(y)

    for d in range(len(Data_x_norm)):
        y = n1.forward([Data_x_norm[d]])
        costo = Costo(y, Data_y_norm[d])
        n1.update_parameters(costo, tz)

    Error_total = perdida_cuadratica_media(costo_d)
    print(f"epoca: {epoch}, error total: {Error_total}")
    if visualizar_graficas:
        error_epoca.append(Error_total)
        epoca_error.append(epoch)
        pesos_error.append(n1.w)

    if epoch % int(epocas/epoca_revision) == 0: # --Visualizar los resultados de la neurona 
        if visualizar_graficas:
            plt.plot(Data_x_norm, Data_y_norm, label='y = x1.8 + 32, (funcion C° a F°)', color='blue')
            plt.plot(Data_x_norm, prediccion_neurona, label=f'y = x{(n1.w[0]*max(data_y))/(max(data_x)):.3f} + {n1.b*max(data_y):.3f}, (funcion del perpectron)', color='red')
            plt.title(f"Funcion de la neurona con respecto a la funcion de C° A F°, en la epoca: {epoch}")
            plt.xlabel("grados C°")
            plt.ylabel("grados F°")
            plt.legend()
            plt.grid(True)
            plt.show()

if visualizar_graficas:
    # --Visualizar en grafica como disminuye el error de la neruona
    plt.plot(epoca_error, error_epoca)
    plt.title("Error con respecto a las epocas")
    plt.xlabel("Epocas")
    plt.ylabel("Error")
    plt.show()

    # --Visualizar en grafica como va cambiando el error cuando el peso w cambia crenado una curva de error
    plt.scatter(error_epoca, pesos_error)
    plt.title("Error con respecto al peso")
    plt.xlabel("Valor de w")
    plt.ylabel("Error")
    plt.show()

# --Comprobar que la red ha aprendido correctamente--
print('-'*64)
Datos_prueba = [5, 15, 65, 34, 90]
for indice in Datos_prueba:
    prediccion_f = n1.forward([indice/max(data_x)])*max(data_y) # La salida del forward se multiplica por el valor maximo de los datos de entrada ya que la nerona trabajo con los datos normalizados
    real = (indice * 1.8) + 32
    print(f"prediccion: {prediccion_f:.3f}, Real: {real}")

print(f"peso w del perceptron: {n1.w}")
print(f"bia b  del perceptron: {n1.b}")


epocas = 100
tz = 0.5

for e in range(epocas+1):

    costos = []

    for d in range(len(Data_x_norm)):
        y = n1.forward([Data_x_norm[d]])
        costo = Costo(y, Data_y_norm[d])
        costos.append(costo)

    for d in range(len(Data_x_norm)):
        y = n1.forward([Data_x_norm[d]])
        costo = Costo(y, Data_y_norm[d])
        n1.update_parameters(costo, tz)

    loss = perdida_cuadratica_media(costos)
    print(f"Perdida: {loss}, epocas:{e}")


























