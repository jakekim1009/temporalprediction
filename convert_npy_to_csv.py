import numpy as np

uganda_locs = np.load("uganda_cluster_locs.npy")

with open('uganda_2011_cluster_locs.csv', 'w') as savefile:
    savefile.write('system:index,lat,lon,.geo\n')
    for i in range(uganda_locs.shape[0]):
        savefile.write(str(i) + ',' + str(uganda_locs[i,0]) + ',' + str(uganda_locs[i,1]) + ',\n')


