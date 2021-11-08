# -*- coding: utf-8 -*-
#���� ����� ����
import numpy as np

#�̹��� �����ֱ� ����
import matplotlib.image as img 

#�׷��� ǥ�ø� ����
import matplotlib.pyplot as plt

#pdf�� �����ϱ� ���� 
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