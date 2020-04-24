from hurricaneObj import Hurricane      #import hurricane class
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
import csv
import itertools

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

# =====================================================================================
# Number of Hurricanes that hit LAND

landfall_hurr = []

for obj in atlantic_hurr_list:
    for recid in obj.recordIdentifier:
        if str(recid).strip() == "L":
            landfall_hurr.append(obj)
            break
        else:
            continue
        # end if
    # end loop
# end loop
# =====================================================================================

# =====================================================================================
# Calculate number of hurricanes of a certain type - Tropical Storm, Hurricane, Other

hu_status = []
ts_status = []
other_status = []

for obj in atlantic_hurr_list:
    ts_bool = False
    hu_bool = False
    other_bool = False

    for status in obj.stormStatus:
        if str(status).strip() == "TS":
            ts_bool = True
        
        elif str(status).strip() == "HU":
            hu_bool = True
            break
        else:
            other_bool = True
    # end loop

    if hu_bool:
        hu_status.append(obj)

    elif ts_bool and not hu_bool:
        ts_status.append(obj)

    else:
        other_status.append(obj)
# end loop


# =====================================================================================

# =====================================================================================
# Calculate number of hurricanes with <10 pts,  10-20pts, 21-30pts, 31-40pts, 41+ pts

less_ten = []
ten_twenty = []
twenty_thirty = []
thirty_forty = []
forty_plus = []
num_points_arr = []

for obj in atlantic_hurr_list:
    if int(obj.numBestTrack) < 10:
        less_ten.append(obj)
    # END IF

    elif int(obj.numBestTrack) > 10 and int(obj.numBestTrack) < 21:
        ten_twenty.append(obj)
    # END ELIF

    elif int(obj.numBestTrack) > 20 and int(obj.numBestTrack) < 31:
        twenty_thirty.append(obj)
    # END ELIF

    elif int(obj.numBestTrack) > 30 and int(obj.numBestTrack) < 41:
        thirty_forty.append(obj)
    # END ELIF

    elif int(obj.numBestTrack) > 40:
        forty_plus.append(obj)
    # END ELIF
# END LOOP
# =====================================================================================


# =====================================================================================
# START MAIN

running = True
while(running):
    print("Type \"1\" for Histogram - # of hurricane data points")
    print("Type \"2\" to print general statistics")
    print("Type \"3\" for Histogram - hurricane types")
    print("Type \"q\" to quit.")
    val = input("INPUT: ")

    if val == "q":
        running = False
    # END IF

    elif val == "1":
        
        x_values = ('less than 10', '10-20', '21-30', '31-40', 'more than 40')
        y_pos = np.arange(len(x_values))
        y_values = [len(less_ten), len(ten_twenty), len(twenty_thirty), len(thirty_forty), len(forty_plus)]
        
        bars = plt.bar(y_pos, y_values, align='center')
        plt.xticks(y_pos, x_values)
        plt.ylabel('# Hurricanes', color='red')
        plt.xlabel('# data entries per hurricane', color='red')
        plt.title('Available Data Entries per Hurricane')

        for bar in bars:
            yh = bar.get_height()
            plt.text(bar.get_x() + 0.01, yh + 3, str(yh), color='black', fontweight='bold')

        plt.show()
    # END ELIF
    
    elif val == "2":
        print('=============================================')
        print('General Info - HURDAT Atlantic Dataset')
        print('=============================================')
        print("Total # hurricanes: " + str(len(atlantic_hurr_list)))
        print("Earliest Hurricane: Amy 1975")
        print("Latest Hurricane: Oscar 2018")
        print("# of Hurricanes that made LANDFALL: " + str(len(landfall_hurr)))
        print('=============================================')
    # END ELIF

    elif val == "3":
        label1 = "Other: Tropical cyclone with a maximum one-minute sustained wind speed of less than 34 knots (40 mph)"
        label2 = "Tropical Storm: Tropical cyclone with a maximum one-minute sustained wind speed of 34 knots(40 mph) to 63 knots (73 mph)"
        label3 = "Hurricane: Tropical cyclone with a maximum one-minute sustained wind speed of 64 knots (74 mph) and above"
        x_vals = ('Other(<34kts)', 'Tropical Storm(34-63kts)', 'Hurricane(>63kts)', )
        y_pos2 = np.arange(len(x_vals))
        y_vals = [len(other_status), len(ts_status), len(hu_status)]

        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3

        bars2 = plt.bar(y_pos2, y_vals, align='center')
        plt.xticks(y_pos2, x_vals)
        plt.ylabel('# Hurricanes', color='red')
        plt.xlabel('Hurricane Status', color='red')
        plt.legend(handles, (label1, label2, label3), loc='upper left', fancybox=True, fontsize='small', handlelength=0, handletextpad=0)
        plt.title('# Hurricanes Per Type')

        for bar in bars2:
            yh = bar.get_height()
            plt.text(bar.get_x() + 0.01, yh + 0.1, str(yh), color='black', fontweight='bold')

        plt.show()

# END WHILE LOOP
# =====================================================================================

# Crop the graph from the beginning, automatically
# plot wind radii
# linear regression
# collect statistics on the dataset (how many hurricanes/data points) - do with a histogram.