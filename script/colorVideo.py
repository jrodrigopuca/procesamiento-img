#!/usr/bin/env python
# coding: utf-8
import cv2, math, sys, getopt, os
import numpy as np 
import matplotlib.pyplot as plt

def changeColor(imagen):
    alto, ancho, cantCanales = imagen.shape
    for x in range(alto):
        for y in range(ancho):
            #Cambia azul
            imagen.itemset((x,y,0),imagen.item(x,y,0))
            #Cambia verde
            imagen.itemset((x,y,1),0)
            #Cambia rojo
            imagen.itemset((x,y,2),0)
    return imagen

def capturarFrames(filename,guardar):
    if guardar:
        path=os.path.dirname(os.path.abspath(filename))
        os.mkdir(path+'/img-'+filename)
    video= cv2.VideoCapture(filename)
    i=0
    success, frame = video.read()
    while success: 
        i+=1
        if guardar:
            nombre = 'img-'+filename+'/'+str(i)+'.jpg'
            cv2.imwrite(nombre, frame) 
        success, frame = video.read()
    return i

def reconstruir(filename, inicio, fin):
    video= cv2.VideoCapture(filename)
    
    fps=video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec =cv2.VideoWriter_fourcc(*'XVID')

    nuevoNombre,extension= filename.split(".")
    nuevoNombre+="-"+str(fin)+'.avi'
    nuevoVideo= cv2.VideoWriter(nuevoNombre,codec,fps,size)   
    i=0
    success, frame = video.read()
    while success:
        if i>=inicio and i<=fin:
            nuevoVideo.write(changeColor(frame))            
        i+=1
        success, frame = video.read()


if __name__=="__main__":
    file='entrada.mp4'
    cantFrames=capturarFrames(file,False)
    #los valores que producen cortes
    reconstruir(file, 0, cantFrames)        