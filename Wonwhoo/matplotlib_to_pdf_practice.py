# -*- coding: utf-8 -*-

#for calculation
import numpy as np
#image
import matplotlib.image as img 

#for graph printout
import matplotlib.pyplot as plt

#for graph table
import pandas as pd

def matrixMult(A):
    row=len(A)
    col=len(A[0])    
    
    B = [[0 for row in range(row)]for col in range(col)]
    
    for i in range(row):
        for j in range(col):
            B[j][i]=A[i][j]
    return B


"""
#for saving in pdf format
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('multipage.pdf')
plt.savefig(pp, format='pdf')
pp.savefig()
pp.close()
"""

#a4 용지 크기는 8.27 inch x 11.69 inch
fig = plt.figure(figsize=(8.27,11.69))
#fig = plt.figure()

plt.title('Benchmark Result',loc = 'left', pad = 30, fontsize = 25)
plt.axis('off')

plt.subplots_adjust(left=0.125, bottom=0.2,  right=0.9, top=0.9, wspace=0.2, hspace=0.35)

img1 = img.imread('Wonwhoo/cherry_image/1.jpg')
axis1 = fig.add_subplot(4,3,1)
axis1.imshow(img1)
axis1.set_title('Succeded Adversarial Examples', loc = 'left', pad = 10)

img2 = img.imread('Wonwhoo/cherry_image/1.jpg')
axis2 = fig.add_subplot(4,3,2)
axis2.imshow(img2)


img4 = img.imread('Wonwhoo/cherry_image/2.jpg')
axis4 = fig.add_subplot(4,3,4)
axis4.imshow(img4)

img5 = img.imread('Wonwhoo/cherry_image/2.jpg')
axis5 = fig.add_subplot(4,3,5)
axis5.imshow(img5)

img7 = img.imread('Wonwhoo/cherry_image/1.jpg')
axis7 = fig.add_subplot(4,3,7)
axis7.imshow(img7)

img8 = img.imread('Wonwhoo/cherry_image/1.jpg')
axis8 = fig.add_subplot(4,3,8)
axis8.imshow(img8)

img10 = img.imread('Wonwhoo/cherry_image/2.jpg')
axis10 = fig.add_subplot(4,3,10)
axis10.imshow(img10)

img11 = img.imread('Wonwhoo/cherry_image/2.jpg')
axis11 = fig.add_subplot(4,3,11)
axis11.imshow(img11)

#adversarial attack result table
axis6 = fig.add_subplot(4,3,6)
axis6.axis('off')
data = [['0.9', '0.4', '0.3'],['0.4', '0.2', '0.1'],['0.8', '0.4', '0.4'],['0.4', '0.4', '0.2']]
data_float = [[0.9, 0.4, 0.3],[0.4, 0.2, 0.1],[0.8, 0.4, 0.4],[0.4, 0.4, 0.2]]
columns = ("Best Case", "Average Case", "Worst Case")
rows = ("FGSM", "PGD", "CW", "DF")

table = axis6.table(
    cellText = data,
    rowLabels= rows,
    colLabels= columns,
    colWidths = [0.5 for x in columns],
    cellLoc='left',
    loc = 'center'
)
data_graph = matrixMult(data_float)
#adversarial attack result graph
axis3 = fig.add_subplot(4,3,3)
axis3.set_title('Attack Results with graph', loc = 'left', pad = 10)
axis3.set_xlabel('FGSM  PGD    CW   DF')
for i in range (0,3):
    axis3.plot(data_graph[i])
axis3.legend(['Best Case', 'Average Case', 'Worst Case'])




axis9 = fig.add_subplot(4,3,9)
axis9.set_title('Advise for model robustness', loc = 'left', pad = 10)
axis9.text(0,0,'Your model is significantly weak\nagainst CW L2 attack, and DeepFool\nattack, But relatively robust against\nFGSM attack, and JSMA attack.\n\nThis weakness can be caused from\n setting hyper parameters, maybe\ninput bias, or input capacity.\n\n',fontsize = 10)
axis9.axis('off')

axis12 = fig.add_subplot(4,3,12)
axis12.text(0,0,'If you think none of this are your\n issues, we recommend adversarial\ntraining with provided our adversarial\nexamples.\n\nTry again with adversarilly trained\nmodel and check out the result.\n\nSee more info in the attached papers.\n\n',fontsize = 10)
axis12.axis('off')

plt.show()

#fig.savefig('multipleplots.pdf')