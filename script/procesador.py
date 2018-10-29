import numpy as np
from scipy import ndimage
import cv2 as cv
import sys, os
import matplotlib.pyplot as plt

#Lista de algunos kernels que andan por ahí

kernelBlur =[[1.0/9,1.0/9,1.0/9],
			[1.0/9,1.0/9,1.0/9],
			[1.0/9,1.0/9,1.0/9]]

kernelGaussianBlur =[[1.0/16,2.0/16,1.0/16],
					[2.0/16,4.0/16,2.0/16],
					[1.0/16,2.0/16,1.0/16]]

kernelGaussianBlurMax =[[1.0/256,4.0/256,6.0/256,4.0/256,1.0/256],
						[4.0/256,16.0/256,24.0/256,16.0/256,1.0/256],
						[6.0/256,24.0/256,36.0/256,24.0/256,6.0/256],
						[4.0/256,16.0/256,24.0/256,16.0/256,1.0/256],
						[1.0/256,4.0/256,6.0/256,4.0/256,1.0/256]]

kernelEdge1= [[1,0,-1],
			[0,0,0],
			[-1,0,1]]

kernelEdge3= [[-1,-1,-1],
			[-1,8,-1],
			[-1,-1,-1]]

kernelLaplacian = [[0,1,0],
				[1,-4,1],
				[0,1,0]]


kernelSharpen= [[0,-1,0],
				[-1,5,-1],
				[0,-1,0]]

kernelSobelX = [[1,0,-1],
				[2,0,-2],
				[1,0,-1]]

kernelSobelY = [[1,2,1],
				[0,0,0],
				[-1,-2,-1]]

kernelPrewittX = [[-1,0,1],
					[-1,0,1],
					[-1,0,1]]

kernelPrewittY = [[1,1,1],
					[0,0,0],
					[-1,-1,-1]]

kernelRobertsX = [[1,0],
				[0,-1]]

kernelRobertsY = [[0,1],
				[-1,0]]

######## Algunas funciones auxiliares #######

#Cargar custom kernel
def cargarKernel(ruta):
	#TODO: Verificar que sea cuadrada + verificar ruta y eso
	archivo_kernel = open (ruta , 'r')
	contenido = [[float(num) for num in line.split(',')] for line in archivo_kernel ]
	#TODO: Parsear para que acepte floats del tipo x/y
	return contenido

#Ayuda
def mostrarAyuda():
	print("\
Para procesar alguna imagen:\n\
\n\
	python procesador.py [nombre_imagen] [opciones] [argumento_extra]\n\
	\n\
	*Las imagenes tienen que estar en el mismo directorio\n\
	\n\
Opciones:\n\
	\n\
	h: Ayuda\n\
	k: Usar kernel desde un archivo, en [argumento_extra] se debe colocar el nombre del mismo.\n\
	b: Blur simple\n\
	gb: Blur Gaussiano 3x3\n\
	gbm: Blur Gaussiano 5x5\n\
	e1: Algun detector de bordes desconocido\n\
	e3: Algun detector de bordes desconocido 2\n\
	l: Laplaciano\n\
	sh: Filtro sharpen\n\
	s: Sobel\n\
	p: Prewitt\n\
	r: Roberts\n\
	mediana: Filtro mediana\n\
	nsp: Ruido sal y pimienta\n\
	ni: Ruido impulsivo\n\
	ng: Ruido Gaussiano\n\
	nr: Rudio Rayleigh\n\n")
	sys.exit()

#Abrir imagen para procesar
def abrir(nombre_imagen):
	directorio_actual = os.getcwd()
	ruta_imagen = directorio_actual + '/' +  nombre_imagen
	imagen = cv.imread(ruta_imagen,0)
	#Chequear si existe
	if imagen is None:
		print("Imagen no encontrada\n")
		mostrarAyuda()
		sys.exit()
	else:
		alto, ancho = imagen.shape
		salida = np.empty([alto,ancho])
		#Hago esto porque opencv o numpy tiene un mecanismo muy gracioso, no puedo copiarlo directamente
		for i in range(alto):
			for j in range(ancho):
				salida[i,j] = imagen[i,j]
		return salida

#Guardar
def guardar(filtro,imagen):
	cv.imwrite(filtro+".png",imagen)

#Guardar y salir
def salir(filtro,imagen):
	cv.imwrite("result/"+filtro+".png",imagen)
	plt.imshow(imagen, cmap='gray')
	sys.exit()

def normalizar(imagen):
	#Normaliza a valores entre 0 y 255
	alto, ancho = imagen.shape
	im_normal = np.empty([alto,ancho])
	maximo = np.amax(imagen)
	minimo = np.amin(imagen)
	shrink = 255/(maximo - minimo)
	return ((imagen - minimo) * shrink).astype(int)


####### Programa principal #######

if sys.argv[1] == "h":
	mostrarAyuda()
else:
	imagen = abrir(sys.argv[1])

#Empezar a procesar
alto, ancho = imagen.shape
filtrada = np.empty([alto,ancho])

#Ver opciones y procesar
#TODO: Mejorar logica de parametros por consola

#Custom kernel
if('k' in sys.argv):
	if len(sys.argv)>3:
		kernelCustom = np.array(cargarKernel(sys.argv[3]))
		filtrada = ndimage.convolve(imagen, kernelCustom).astype(int)
		salir("custom_kernel",filtrada)
	else:
		print("Falta ruta del archivo con el kernel")

if('b' in sys.argv):
	filtrada = ndimage.convolve(imagen, kernelBlur)
	salir("blur",filtrada)

if('gb' in sys.argv):
	filtrada = ndimage.convolve(imagen, kernelGaussianBlur)
	salir("gaussian_blur",filtrada)

if('gbm' in sys.argv):
	filtrada = ndimage.convolve(imagen, kernelGaussianBlurMax)
	salir("gaussian_blur_max",filtrada)

if('e1' in sys.argv):
	filtrada = ndimage.convolve(imagen, kernelEdge1)
	salir("edge1",filtrada)

if('e3' in sys.argv):
	filtrada = ndimage.convolve(imagen, kernelEdge3)
	salir("edge3",filtrada)

if('l' in sys.argv):
	filtrada = ndimage.convolve(imagen, kernelLaplacian)
	salir("laplacian",filtrada)

if('sh' in sys.argv):
	filtrada = ndimage.convolve(imagen, kernelSharpen)
	salir("sharpen",filtrada)


##### Edge detection mas trabajada: Sobel, Prewitt y Roberts

if('s' in sys.argv):
	umbral = 30
	if len(sys.argv)>3:
		umbral = int(sys.argv[3])
	derivadax = np.empty([alto,ancho])
	derivadax = ndimage.convolve(imagen, kernelSobelX) 
	derivaday = np.empty([alto,ancho])
	derivaday = ndimage.convolve(imagen, kernelSobelY)
	#Normalizar antes de guardar
	guardar("sobel_x",normalizar(derivadax))
	guardar("sobel_y",normalizar(derivaday))
	filtrada = normalizar(np.hypot(derivadax,derivaday))
	salir("sobel" + "-" + str(umbral),((filtrada>umbral) * filtrada))

if('p' in sys.argv):
	umbral = 30
	if len(sys.argv)>3:
		umbral = int(sys.argv[3])
	derivadax = np.empty([alto,ancho])
	derivadax = ndimage.convolve(imagen, kernelPrewittX) 
	derivaday = np.empty([alto,ancho])
	derivaday = ndimage.convolve(imagen, kernelPrewittY)
	#Normalizar antes de guardar
	guardar("prewitt_x",normalizar(derivadax))
	guardar("prewitt_y",normalizar(derivaday))
	filtrada = normalizar(np.hypot(derivadax,derivaday))
	salir("prewitt" + "-" + str(umbral),((filtrada>umbral) * filtrada))

if('r' in sys.argv):
	umbral = 30
	if len(sys.argv)>3:
		umbral = int(sys.argv[3])
	derivadax = np.empty([alto,ancho])
	derivadax = ndimage.convolve(imagen, kernelRobertsX) 
	derivaday = np.empty([alto,ancho])
	derivaday = ndimage.convolve(imagen, kernelRobertsY)
	#Normalizar antes de guardar
	guardar("roberts_x",normalizar(derivadax))
	guardar("roberts_y",normalizar(derivaday))
	filtrada = normalizar(np.hypot(derivadax,derivaday))
	salir("roberts" + "-" + str(umbral),((filtrada>umbral) * filtrada))


##### Filtro mediana

if('mediana' in sys.argv):
	#Hice un filtro de mediana con opencv, numpy y un filtro de mediana de scipy
	#Por alguna razon es bastante lento (porque tiene que ordenar 9 elementos por cada pixel)
	salir("mediana",ndimage.median_filter(imagen, 3))


##### Ruidos

#TODO: no iterar para detectar los umbrales en ruido impulsivo y sal-pimienta
if('nsp' in sys.argv):
	#Condimentar
	umbralSal = 0.999
	umbralPimienta = 0.0001
	if len(sys.argv)>3:
		#Nos da 1000 valores de umbral en total (si solo usamos enteros)
		umbralSal = (1000-float(sys.argv[3]))/1000
		umbralPimienta = (float(sys.argv[3]))/1000
	for i in range(alto):
			for j in range(ancho):
				valor = np.random.random_sample()
				if valor>umbralSal:
					filtrada[i,j] = 255
				elif valor<umbralPimienta:
					filtrada[i,j] = 0
				else:
					filtrada[i,j] = imagen[i,j]
	salir("ruido_sal-pimienta" + "-" + str(umbralSal) + "-" + str(umbralPimienta),filtrada)

if('ni' in sys.argv):
	#Salar
	umbral = 0.999
	if len(sys.argv)>3:
		#Nos da 1000 valores de umbral en total (si solo usamos enteros)
		umbral = (1000-float(sys.argv[3]))/1000
	for i in range(alto):
			for j in range(ancho):
				if np.random.random_sample()>umbral:
					filtrada[i,j] = 255
				else:
					filtrada[i,j] = imagen[i,j]
	salir("ruido_impulsivo" + "-" + str(umbral),filtrada)

if('ng' in sys.argv):
	#Ruido gaussiano
	media = 0
	desviacion = 1
	if len(sys.argv)>3:
		desviacion = float(sys.argv[3])
	salir("ruido_gaussiano" + "-" + str(media) + "-" + str(desviacion),\
		(imagen + np.random.normal(media,desviacion,(alto,ancho))))

if('nr' in sys.argv):
	#Ruido Rayleigh
	escala = 1
	if len(sys.argv)>3:
		escala = float(sys.argv[3])
	salir("ruido_rayleigh" + str(escala),imagen + np.random.rayleigh(escala,(alto,ancho)))