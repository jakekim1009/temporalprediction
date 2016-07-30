import numpy as np
import os

from tifffile import *

#export_dir = '/Users/jakekim/PycharmProjects/historical_imagery/exports/'
#export_dir = '/Users/jakekim/PycharmProjects/historical_imagery/uganda_2011/'
#export_dir = '/Users/jakekim/PycharmProjects/historical_imagery/'
#export_files = os.listdir(export_dir)
#export_files = [export_dir + export_file for export_file in export_files]

#export_files = ['/Users/jakekim/PycharmProjects/historical_imagery/exports/UGANDA_ROW5_LON31.93_LAT0.64_TEST.tif']

export_files = ['/Users/jakekim/PycharmProjects/historical_imagery/uganda_2012_testcomposite_1.tif']

#fn = 'UGANDA_ROW1_LON31.58_LAT0.57_TEST'
for export_file in export_files:
    if export_file[-4:] != '.tif': continue
    with TiffFile(export_file) as tif:
        images = tif.asarray()

        # Slice the B,G,R channels and transpose so that the color channels come to the front.
        RGB_image = images[:,:,0:3].transpose(2,0,1)
        # Change rows so that  BGR -> RGB.
        RGB_image[[0,2],:] = RGB_image[[2,0],:]
        imsave(export_file[:-4] + '_210' + '.tif', RGB_image)
