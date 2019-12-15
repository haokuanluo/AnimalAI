from PIL import Image
import numpy
im = Image.open("crr/my73.png")
np_im = numpy.array(im)
print (np_im[40:60:2,:60:2,0])

print (np_im[40:60:2,:60:2,1])

print (np_im[40:60:2,:60:2,2])