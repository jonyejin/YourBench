# -*- coding: utf-8 -*-
#수식 계산을 위함
import numpy as np

#이미지 보여주기 위함
import matplotlib.image as img 

#그래프 표시를 위함
import matplotlib.pyplot as plt

#pdf로 저장하기 위함 
from matplotlib.backends.backend_pdf import PdfPages
"""
pp = PdfPages('multipage.pdf')
plt.savefig(pp, format='pdf')
pp.savefig()
pp.close()
"""

fig = plt.figure()
axis1_1 = fig.add_subplot(2,1,1)
axis1_1.plot(range(10))
axis1_2 = fig.add_subplot(2,1,2)
axis1_2.plot(range(10,20))
fig.savefig('multipleplots.png')
"""
X = np.linspace(0,np.pi,100)
Y = np.sin(X)
plt.axes(polar=True)
plt.plot(X,Y)
plt.show()

image = img.imread('cherry_image/1.jpg')
plt.imshow(image)
plt.show()
"""