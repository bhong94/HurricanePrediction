from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon(list(xy), edgecolor='black', alpha=0.2 )
    plt.gca().add_patch(poly)

# takes bottom right and top left point of the square
def FindPoint(x1, y1, x2, y2, x, y) : 
    if (x <= x1 and x >= x2 and y >= y1 and y <= y2) : 
        return True
    else : 
        return False    

def GetCenter(x1, y1, x2, y2):
    x = (x1+x2)/2
    y = (y1+y2)/2
    return x, y

# min(0,0) max(110,70)

# Create 1x1 grid
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

print(grid[7699])

# for i in range(70):
#     for j in range(110):
#         grid[cellNum][j]

map = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=70,\
    llcrnrlon=-110,urcrnrlon=0,resolution='c')
# br, tr, tl, bl
lats = [30, 35, 35, 30]
lons = [-60, -60, -65, -65]
map.drawcoastlines()
map.drawcountries(color='red')
map.drawstates(color='darkblue')
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgreen', lake_color='aqua')

xlon = 109.2
ylat = 69.5

# x,y = map(xlon*-1, ylat)

# check which cell a point falls in
for cell in grid:
    if FindPoint(cell[0][0]*-1, cell[0][1], cell[2][0]*-1, cell[2][1], xlon*-1, ylat):
        cellHit = cell
        print(cellHit)

# inside = FindPoint(-109.0, 69.0, -110.0, 70.0, -109.2, 69.5)
# print(inside)

dx, dy = GetCenter(grid[20][0][0]*-1, grid[20][0][1], grid[20][2][0]*-1, grid[20][2][1])
x,y = map(dx, dy)

# draw the point
map.plot(x, y, 'o-', markersize=1, c='red', label='Observation')


# inside = FindPoint(lons[0], lats[0], lons[2], lats[2], -69, 33)
# print(inside)
# x,y = map(-69,33)


# draw the grid
# for cell in grid:
#     cellLons = [cell[0][0]*-1, cell[1][0]*-1, cell[2][0]*-1, cell[3][0]*-1]
#     cellLats = [cell[0][1], cell[1][1], cell[2][1], cell[3][1]]
#     draw_screen_poly(cellLats, cellLons, map)


# map.plot(x, y, 'o-', markersize=2, c='red', label='Observation')

parallels = map.drawparallels(np.arange(0,70,1), dashes=[1,0], linewidth=0.5)
meridians = map.drawmeridians(np.arange(-110,0,1), dashes=[1,0], linewidth=0.5)


plt.show()