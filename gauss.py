from hurricaneObj import Hurricane
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools
import time
import re

# selects first result of the search
def first(iterable, default=None):
  for item in iterable:
    return item
  return default
# END METHOD

# returns random 'n%' of elements from 2d array(exept first and last elements)
# n = % training data (i.e if n=0.2, returns 20% of elements from the array)
def makeTrainArrays(arr2d, arr, n):
    num = 1-n
    numElements = int(len(arr)*num)
    for i in range(numElements):
        random_index = np.random.randint(1, len(arr2d)-1)
        arr2d = np.delete(arr2d, random_index, 0)
        arr = np.delete(arr, random_index, 0)
    
    return arr2d, arr
# END METHOD

temp_mem = []
atlantic_hurr_list = []

#Process atlantic hurricane data set
with open('hurdat_short.csv', 'r') as csv_file:
    full_atlantic_data = csv.reader(csv_file)
    for line in full_atlantic_data:
        temp_mem.append(line)         
# END WITH OPEN

index = 0
for line in temp_mem:
    if len(line) <= 4:
        hurricane_obj = Hurricane(line[0], line[1], line[2])

        best_track_idx = index + 1
        max_idx = best_track_idx + (int(line[2]) - 1)
        while best_track_idx <= max_idx:
            best_track_entry = temp_mem[best_track_idx]

            hurricane_obj.dateTime.append(best_track_entry[0])
            hurricane_obj.recordIdentifier.append(best_track_entry[2])
            hurricane_obj.stormStatus.append(best_track_entry[3])
            hurricane_obj.latitude.append(best_track_entry[4])
            hurricane_obj.longitude.append(best_track_entry[5])
            hurricane_obj.maxSustWind.append(best_track_entry[6])
            hurricane_obj.minPressure.append(best_track_entry[7])
            hurricane_obj.wr34_northEast.append(best_track_entry[8])
            hurricane_obj.wr34_southEast.append(best_track_entry[9])
            hurricane_obj.wr34_southWest.append(best_track_entry[10])
            hurricane_obj.wr34_northWest.append(best_track_entry[11])
            hurricane_obj.wr50_northEast.append(best_track_entry[12])
            hurricane_obj.wr50_southEast.append(best_track_entry[13])
            hurricane_obj.wr50_southWest.append(best_track_entry[14])
            hurricane_obj.wr50_northWest.append(best_track_entry[15])
            hurricane_obj.wr64_northEast.append(best_track_entry[16])
            hurricane_obj.wr64_southEast.append(best_track_entry[17])
            hurricane_obj.wr64_southWest.append(best_track_entry[18])
            hurricane_obj.wr64_northWest.append(best_track_entry[19])

            best_track_idx += 1     # update index for inner loop
        # END WHILE LOOP
        
        # append hurricane to list of objects
        atlantic_hurr_list.append(hurricane_obj)

    # END IF
    
    # update the index for outer loop
    index += 1      

# END FOR IN
# ======================================================================================

# ======================================================================================
# Search for hurricane
print('============================')
print("Enter Hurricane ID")
print('============================')
inp = input("INPUT: ")

select = first(hurr for hurr in atlantic_hurr_list if hurr.id == inp)
# ======================================================================================

# ======================================================================================
# Grab coordinates
# y_actual = np.array([])
# for lat in select.latitude:
#     y_actual = np.append(y_actual, float(lat.replace('N', '')))
# END LOOP

y_actual = np.array([])
x_actual = np.array([])
for i in range(int(select.numBestTrack)):
    if not re.search('[a-zA-Z]', select.recordIdentifier[i]):
        x_actual = np.append(x_actual, float(select.longitude[i].replace('W', '')))
        y_actual = np.append(y_actual, float(select.latitude[i].replace('N', '')))


# x_actual = np.array([])
# for lon in select.longitude:
#     x_actual = np.append(x_actual, float(lon.replace('W', '')))
# END LOOP

# Join arrays
coordinates_actual = np.arange(len(x_actual)*2).astype(float).reshape(len(x_actual),2)
for i in range(len(x_actual)):
    coordinates_actual[i][0] = float(x_actual[i])
    coordinates_actual[i][1] = float(y_actual[i])

# ======================================================================================
# x_pred = np.atleast_2d(np.linspace(x_actual[0], x_actual[x_actual.size-1], 100)).T

print(coordinates_actual.shape)

# split data for training
# x = np.delete(x_actual, np.arange(0, x_actual.size, 2))
# y = np.delete(y_actual, np.arange(0, y_actual.size, 2))

# X = x
# X = X.reshape(-1,1)
# x_actual = x_actual.reshape(-1,1)

# coordinates_actual = np.delete(coordinates_actual, np.arange(1, coordinates_actual.size, 2))

# Prepare time array - "Hours elapsed since start of hurricane"
hours_elapsed = 0
time_actual = np.array([])
for i in range(len(coordinates_actual)):
    if i == 0:
        time_actual = np.append(time_actual, int(hours_elapsed))
    else:
        hours_elapsed += 6
        time_actual = np.append(time_actual, int(hours_elapsed))    


# time_actual = (np.arange(len(x_actual)))
time_actual = time_actual.reshape(-1,1)

coordinates_train, time_train = makeTrainArrays(coordinates_actual, time_actual, 0.2)

kernel = RBF(length_scale=10, length_scale_bounds=(1e-3, 1e3))
m = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
m.fit(time_train, coordinates_train)
# r = m.score(X, y)
y_pred, sigma = m.predict(time_actual, return_std=True)

print(sigma)
# print(X.shape)
print(y_pred)

# Convert y_pred to separate arrays to be used for plotting
lon_pred = np.array([])
lat_pred = np.array([])
for i in range(len(y_pred)):
    lon_pred = np.append(lon_pred, float(y_pred[i][0]))
    lat_pred = np.append(lat_pred, float(y_pred[i][1]))

# print the predicted trajectory of the hurricane
print("predicted trajectory")
fig = plt.figure(figsize=(12,9))

map = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=60,\
    llcrnrlon=-150,urcrnrlon=0,resolution='c')


xpred, ypred = map(lon_pred*-1, lat_pred)
xact, yact = map(x_actual*-1, y_actual)
map.plot(xpred, ypred, 'o-', markersize=3, linewidth=1, c='r', label='Prediction')
map.plot(xact, yact, '-', linewidth=2, c='blue', label='Observation')

map.drawcoastlines()
map.drawcountries(color='red')
map.drawstates(color='darkblue')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgreen', lake_color='aqua')

map.drawparallels(np.arange(-90,90,10), labels=[1,0,0,0])
map.drawmeridians(np.arange(-180,180,10), labels=[0,0,0,1])

# Handle Confidence Interval
plt.fill(np.concatenate([xpred, xpred[::-1]]), np.concatenate([ypred - 1.9600 *100000* (sigma+1), (ypred + 1.9600 *100000* (sigma+1))[::-1]]),
    alpha=.5, fc='black', ec='None', label='95% confidence interval')


plt.legend(loc='upper left')
# show map
plt.title('Hurricane Trajectory - Gaussian Process', fontsize=20)
plt.show()