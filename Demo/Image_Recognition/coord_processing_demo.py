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
    v2 = P3 - P2

    v1_u = unit_vector(v1).flatten() #aplsatamos para que se pueda hacer dot product
    v2_u = unit_vector(v2).flatten()
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) # output en rad

# Recorrer con un bucle todos los frames
max_frames = coordinates.loc[:,"frame"].max(axis = 0)

# PX = np.array(df.loc[i,['x','y','z']]) donde i es el landmark
# angle_between(PX,PX,PX)

# Generamos una lista de las landmarks que habrá que restar para obtener los angulos
list_points = [
    [1,2,3], #alfa_I
    [2,3,4], #alfa_II
    [5,6,7], #beta_I
    [6,7,8], #beta_II
    [9,10,11], #gamma_I
    [10,11,12], #gamma_II
    [13,14,15], #lambda_I
    [14,15,16], #lambda_II
    [17,18,19], #epsilon_I
    [18,19,20]] #epsilon_II

for frame in range(max_frames):
    angle_coordinates.at[frame,'frame'] = frame + 1
    changing_angle = 1 #contador para ir cambiando los angulos en el siguiente bucle
    coordinates_at_frame = coordinates.loc[coordinates['frame'] == frame + 1] #cribamos a los datos de este frame

    for point in list_points:
        P1 = np.array(coordinates_at_frame.loc[coordinates_at_frame['landmark_id'] == point[0],['x','y','z']]) #esto funciona porque dios quiere
        P2 = np.array(coordinates_at_frame.loc[coordinates_at_frame['landmark_id'] == point[1],['x','y','z']])
        P3 = np.array(coordinates_at_frame.loc[coordinates_at_frame['landmark_id'] == point[2],['x','y','z']])

        angle_coordinates.iat[frame,changing_angle] = angle_between(P1,P2,P3)
        changing_angle = changing_angle + 1