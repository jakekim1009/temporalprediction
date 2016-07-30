from geotiling import ImageComposite
import ee
#ee.Initialize()


z = ImageComposite('./uganda_2012_testcomposite_1.tif')
z.getgridwins('./uganda_2011_cluster_locs.csv', 2, 1, 333, 333, 'UGANDA', '', './uganda_2011')

