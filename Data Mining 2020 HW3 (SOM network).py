import numpy as np
import pandas as pd
import random
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
np.random.seed(321)
random.seed(321)
    
def shuffle(x):
    list_ = list(np.arange(0, len(x)).astype(int))
    random.shuffle(list_)
    return x.reindex(list_)

def normalization(X):
    return (X- np.min(X))/ (np.max(X)- np.min(X))

def coordinate_value(map_size, figure_show_up):
    
    location_index= np.arange(map_size- 1,  -1, -1)
    
    list_value= []
    for x in range(0, int(map_size)):
        list_in= []
        list_value.append(list_in)
        for y in range(0, int(map_size)):
            list_in.append((location_index[x], location_index[y]))
            
    coordinate_value= pd.DataFrame(index= np.arange(map_size- 1,  -1, -1))
    for location in range(0, len(list_value)):
        coordinate_value[str(location)]= list_value[location]

    coordinate_value.columns= np.arange(map_size- 1,  -1, -1)
    coordinate_value= coordinate_value.reindex(columns= np.arange(0, map_size))
    
    if figure_show_up== True:

        X_data= []; y_data= []
        for data in np.array(coordinate_value).ravel():
            X_data.append(data[0]); y_data.append(data[1])
    
        plt.figure(num= 1, figsize= (5, 5))
        plt.scatter(x= X_data, y= y_data, c= "blue", marker= "o", label= "Topology Coordinate")
        plt.xlabel("X", fontsize= 10)
        plt.ylabel("y", fontsize= 10)
        plt.grid(True)
        plt.legend(loc= "best")
        plt.show()

    return np.array(coordinate_value)

def euclidean_distance(X, y):
    return np.sum((np.array(X)- np.array(y))** 2)** 0.5

def neighborhood(neurons, neighborhood_radius, topology_coordinate, winner_location):
    R= neighborhood_radius
    topology_coordinate= np.array(topology_coordinate)
    
    neighborhood_distance= []
    for i in range(0, topology_coordinate.shape[0]):
        for j in range(0, topology_coordinate.shape[1]):
            x= (topology_coordinate[i][j][0]- topology_coordinate[winner_location][0][0])** 2
            y= (topology_coordinate[i][j][1]- topology_coordinate[winner_location][0][1])** 2
            
            neighborhood_distance.append((x+ y)** 0.5)
    
    neighborhood_distance= np.exp((-1* np.array(neighborhood_distance))/ R).reshape(int(neurons** 0.5), int(neurons** 0.5))
    return neighborhood_distance


def SOM_Neural_Network(X_in, Neurons, Eta, Eta_rate, R_radius, R_radius_rate, Iteration):
    n, dimensions= X_in.shape
    topology_coordinate= coordinate_value(map_size= Neurons** 0.5, figure_show_up= False)
    output_layer= np.zeros(shape= (int(Neurons** 0.5), int(Neurons** 0.5)))
    W_matrix= np.random.random(size= (dimensions, Neurons))
    total_distance_list= []
    for iteration in range(0, Iteration):
        total_distance= []
        for n in range(0, n):
            net_j_k= []
            for i in range(0, Neurons):
                node= np.sum((X_in[n, :]- W_matrix.T[i, :])** 2)
                net_j_k.append(node)
            winning_node= np.where(np.array(net_j_k).reshape(int(Neurons** 0.5),
                                                             int(Neurons** 0.5))== np.min(np.array(net_j_k).reshape(int(Neurons** 0.5),
                                                                                                                    int(Neurons** 0.5))))
            neighborhood_distance= neighborhood(Neurons, R_radius, topology_coordinate, winning_node)
            if (iteration+ 1)== Iteration:
                wining_node= np.where(np.array(net_j_k).reshape(int(Neurons** 0.5),
                                                                int(Neurons** 0.5))== np.min(np.array(net_j_k).reshape(int(Neurons** 0.5),
                                                                                                                       int(Neurons** 0.5))))
                output_layer[wining_node]+= 1
            distance= []
            for update in range(0, Neurons):
                W_matrix.T[update, :]+= Eta* (X_in[n, :]- W_matrix.T[update, :])* neighborhood_distance.ravel()[update]                
                distance.append((np.sum((X_in[n, :]- W_matrix.T[update, :])** 2)** 0.5))

            distance_p= np.min(distance)
            total_distance.append(distance_p)
            
        total_distance_list.append(np.sum(total_distance))
        if iteration% 10== 0:  
            print("Iteration= %4d, Total distance %4.4f"% (iteration, np.sum(total_distance)))
        Eta= Eta* Eta_rate
        R_radius= R_radius* R_radius_rate
    return output_layer, np.array(total_distance_list)

### download database
path= "/Users/ericchou/Desktop/PyCharm/Data Mining HW PROGRAM/Iris.csv"

with open(path, "r") as file:
    database= pd.read_csv(file)

database = shuffle(database).drop(["Id", "Species"], axis=1)
database= normalization(database)

X_input= np.array(database)

### hyperparameters
Neurons= 10* 10

Eta= 0.1; Eta_rate= 0.1
R_radius= (Neurons** 0.5)/ 2; R_radius_rate= 0.1

Epoch= 100

output, total_distance= SOM_Neural_Network(X_in= X_input, Neurons= Neurons, Eta= Eta, Eta_rate= Eta_rate, R_radius= R_radius, R_radius_rate= R_radius_rate, Iteration= Epoch)

plt.figure(num= 1, figsize= (9, 4))
plt.title("Error is measured by using ''Total Distance''", fontsize= 13)
plt.plot(np.arange(1, Epoch+ 1), total_distance, label= "Training rate = 0.9")
plt.grid(True)
plt.ylabel("Error", fontsize= 14)
plt.xlabel("Epoch", fontsize= 14)
plt.legend(loc= "best", fontsize= 14)
plt.show()

fig= plt.figure(num= 2, figsize= (6, 6))
ax= fig.gca(projection= "3d")
X = np.arange(0, Neurons** 0.5)
Y = np.arange(0, Neurons** 0.5)
X, Y = np.meshgrid(X, Y)
Z= output

surf= ax.plot_surface(X, Y, Z, rstride= 1, cstride= 1, cmap= cm.coolwarm, linewidth= 0, antialiased= False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink= 0.5, aspect= 5)
plt.show()