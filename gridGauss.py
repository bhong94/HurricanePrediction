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
#try joyce2018
# ======================================================================================
# selects first result of the search
def first(iterable, default=None):
  for item in iterable:
    return item
  return default
# END METHOD
# ======================================================================================
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
# ======================================================================================
# draw grid cells
def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon(list(xy), edgecolor='black', alpha=0.2 )
    plt.gca().add_patch(poly)
# END METHOD
# ======================================================================================
# Checks if a given point falls inside the area of a grid cell
# takes bottom right and top left points of the square
def FindPoint(x1, y1, x2, y2, x, y) : 
    if (x <= x1 and x >= x2 and y >= y1 and y <= y2) : 
        return True
    else : 
        return False 
# END METHOD
# ======================================================================================
# check which cell a point falls in
def FindGridCell(gridBoard, lonPoint, latPoint):
    idx = 0    
    for cell in gridBoard:
        if FindPoint(cell[0][0]*-1, cell[0][1], cell[2][0]*-1, cell[2][1], lonPoint*-1, latPoint):
            return cell, idx
        # else:
        #     cellHit = False
        idx += 1
# ======================================================================================
def GetCenter(x1, y1, x2, y2):
    x = (x1+x2)/2
    y = (y1+y2)/2
    return x, y
# ======================================================================================
# Grab data from the dataset
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
# Create 1x1 grid - 1x1 cells based on max and min lat/lon coordinates out of all the hurricanes
brLon = 0.0
brLat = 0.0

grid = np.arange(7700*4*2).astype(float).reshape(7700,4,2)
rowNum = 0
cellNum = 0
for cell in grid:

    cell[0][0] = brLon          # bottom right lon
    cell[0][1] = brLat          # bottom right lat

    cell[1][0] = brLon          # top right lon
    cell[1][1] = brLat + 1.0    # top right lat
    
    cell[2][0] = brLon + 1.0    # top left lon
    cell[2][1] = brLat + 1.0    # top left lat
    
    cell[3][0] = brLon + 1.0    # bottom left lon
    cell[3][1] = brLat          # bottom left lat
    brLon = cell[3][0]
    brLat = cell[3][1]
    cellNum += 1

    # if its a new row, reset
    if cellNum % 110 == 0:
        rowNum += 1
        brLon = 0
        brLat = rowNum
# END FOR LOOP   
# ======================================================================================
# Search for hurricane
print('============================')
print("Enter Hurricane ID")
print('============================')
inp = input("INPUT: ")

select = first(hurr for hurr in atlantic_hurr_list if hurr.id == inp)
# ======================================================================================
# Grab coordinates
y_actual = np.array([])
x_actual = np.array([])
for i in range(int(select.numBestTrack)):
    if not re.search('[a-zA-Z]', select.recordIdentifier[i]):
        x_actual = np.append(x_actual, float(select.longitude[i].replace('W', '')))
        y_actual = np.append(y_actual, float(select.latitude[i].replace('N', '')))


# Join arrays
coordinates_actual = np.arange(len(x_actual)*2).astype(float).reshape(len(x_actual),2)
for i in range(len(x_actual)):
    coordinates_actual[i][0] = float(x_actual[i])
    coordinates_actual[i][1] = float(y_actual[i])

# ======================================================================================
# Get the grids that the trajectory falls in
pathCells = []
pathCellIds = []
for i in range(len(coordinates_actual)):
    cell, idx = FindGridCell(grid, coordinates_actual[i][0], coordinates_actual[i][1])
    pathCells.append(cell)
    pathCellIds.append(idx)

pathCells = np.array(pathCells)
pathCellIds = np.array(pathCellIds)
# print(pathCells.shape)
# print(pathCells)
# print(pathCellIds)
# ======================================================================================
center_points = []
for cll in pathCells:
    cx, cy = GetCenter(cll[0][0]*-1, cll[0][1], cll[2][0]*-1, cll[2][1])
    center_points.append([cx,cy])
center_points = np.array(center_points)
# ======================================================================================
# Prepare time array - "Hours elapsed since start of hurricane"
hours_elapsed = 0
time_actual = np.array([])
for i in range(len(center_points)):
    if i == 0:
        time_actual = np.append(time_actual, int(hours_elapsed))
    else:
        hours_elapsed += 6
        time_actual = np.append(time_actual, int(hours_elapsed))
# ======================================================================================
time_actual = time_actual.reshape(-1,1)
# center_points = center_points.reshape(-1,1)
# ======================================================================================
# coordinates_train, time_train = makeTrainArrays(coordinates_actual, time_actual, 0.2)
cells_train, time_train = makeTrainArrays(center_points, time_actual, 0.2)

# cells_train = np.delete(pathCellIds, np.arange(0, pathCellIds.size, 2)).reshape(-1,1)
# outputCells_train = np.delete(outputCells, np.arange(0, outputCells.size, 2)).reshape(-1,1)



kernel = RBF(length_scale=10, length_scale_bounds=(1e-3, 1e3))
m = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
m.fit(time_train, cells_train)
# r = m.score(X, y)
y_pred, sigma = m.predict(time_actual, return_std=True)

# print(sigma)
# print(X.shape)
# print(pathCellIds)
# print(outputCells)
print(y_pred.shape)

cells_pred = []
ids_pred = []
for i in range(len(y_pred)):
    cell, idx = FindGridCell(grid, y_pred[i][0]*-1, y_pred[i][1])
    cells_pred.append(cell)
    ids_pred.append(idx)

cells_pred = np.array(cells_pred)
ids_pred = np.array(ids_pred)

center_pred = []
for cll in cells_pred:
    cx, cy = GetCenter(cll[0][0]*-1, cll[0][1], cll[2][0]*-1, cll[2][1])
    center_pred.append([cx,cy])
center_pred = np.array(center_pred)

print(sigma)
# print(center_pred)

# Convert center_pred to separate arrays to be used for plotting
lon_pred = np.array([])
lat_pred = np.array([])
for i in range(len(center_pred)):
    lon_pred = np.append(lon_pred, float(center_pred[i][0]))
    lat_pred = np.append(lat_pred, float(center_pred[i][1]))

# ======================================================================================
# print the predicted trajectory of the hurricane
print("predicted trajectory")
fig = plt.figure(figsize=(12,9))

map = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=70,\
    llcrnrlon=-110,urcrnrlon=0,resolution='c')


xpred, ypred = map(lon_pred, lat_pred)
xact, yact = map(x_actual*-1, y_actual)
map.plot(xpred, ypred, 'o-', markersize=3, linewidth=1, c='r', label='Prediction')
map.plot(xact, yact, '-', linewidth=2, c='blue', label='Observation')

map.drawcoastlines()
map.drawcountries(color='red')
map.drawstates(color='darkblue')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgreen', lake_color='aqua')

parallels = map.drawparallels(np.arange(0,70,1), dashes=[1,0], linewidth=0.5)
meridians = map.drawmeridians(np.arange(-110,0,1), dashes=[1,0], linewidth=0.5)

# Handle Confidence Interval
plt.fill(np.concatenate([xpred, xpred[::-1]]), np.concatenate([ypred - 1.9600 *100000* (sigma+1), (ypred + 1.9600 *100000* (sigma+1))[::-1]]),
    alpha=.5, fc='black', ec='None', label='95% confidence interval')


plt.legend(loc='upper left')
# show map
plt.title('Hurricane Trajectory - Gaussian Process', fontsize=20)
plt.show()
# ======================================================================================