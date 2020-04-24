from hurricaneObj import Hurricane      #import hurricane class
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools
from scipy.interpolate import *
from scipy.stats import *

# selects first result of the search
def first(iterable, default=None):
  for item in iterable:
    return item
  return default

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

# Data is processed.
# START LINEAR REGRESSION - Hurricane Isidore 1990

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

y = np.array([])
for lat in select.latitude:
    y = np.append(y, float(lat.replace('N', '')))
# END LOOP

x = np.array([])
for lon in select.longitude:
    x = np.append(x, float(lon.replace('W', '')))
# END LOOP

# ======================================================================================

linFit = np.polyfit(x, y, 1)

y_pred = linFit[0] * x + linFit[1]
slope,intercept,r_value,p_value,std_err = linregress(x,y)

# print the predicted trajectory of the hurricane
print("predicted trajectory")
fig = plt.figure(figsize=(12,9))

map = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=60,\
    llcrnrlon=-150,urcrnrlon=0,resolution='c')

map.drawcoastlines()
map.drawcountries(color='red')
map.drawstates(color='darkblue')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgreen', lake_color='aqua')

map.drawparallels(np.arange(-90,90,10), labels=[1,0,0,0])
map.drawmeridians(np.arange(-180,180,10), labels=[0,0,0,1])

xpred, ypred = map(x*-1, y_pred)
xact, yact = map(x*-1, y)
map.plot(xpred, ypred, 'o-', markersize=3, linewidth=1, c='r', label='Prediction')
map.plot(xact, yact, '-', linewidth=2, c='blue', label='Observation')

# show map
plt.title('Hurricane Trajectory - Linear Regression', fontsize=20)
plt.show()


