import cv2 as cv
import numpy as np

route = 'camiseta2.jpg'

original = cv.imread(route)
alto, ancho, cantCanales = original.shape


for x in range(alto):
	for y in range(ancho):

		if (original.item(x,y,0)>= 250) and (original.item(x,y,1)>= 250) and (original.item(x,y,2)>= 250):
			#Cambia azul
			original.itemset((x,y,0),0)
			#Cambia verde
			original.itemset((x,y,1),0)
			#Cambia rojo
			original.itemset((x,y,2),0)

cv.imwrite('gatiCambioColor.png',original)
cv.imshow('Cambio color',original)
cv.waitKey(0)
cv.destroyAllWindows()