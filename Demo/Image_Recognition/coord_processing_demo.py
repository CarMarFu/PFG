import pandas as pd
import numpy as np

# Leer csv
coordinates = pd.read_csv("./landmarks_demo.csv")

# Crear un nuevo set para guardar las nuevas coordenadas
angle_coordinates = pd.DataFrame(columns=['frame','alfa_I','alfa_II','beta_I','beta_II','gamma_I','gamma_II','lambda_I','lambda_II','epsilon_I','epsilon_II'])

# Funciones necesarias para procesar los ángulos
def unit_vector(vector): # calcular vector unitario
    return vector / np.linalg.norm(vector)

def angle_between(P1,P2,P3): # sacar ángulo con el arccoseno
    v1 = P2 - P1
    v2 = np.array([x3-x2,y3-y2,z3-z2])

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) # output en rad

# Recorrer con un bucle todos los frames
max_frames = coordinates.loc[:,"frame"].max(axis = 0)

# PX = np.array(df.loc[i,['x','y','z']]) donde i es el landmark
# angle_between(PX,PX,PX)