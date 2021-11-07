import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from GQA.utils.plotutils import CMAP,COLOR

# Peaks function class. Chromosome are binary numbers, but peaks is R^2 -> R function so we need to convert binary to real in a precise way. The idea is to bound reals to have a good level of precision.
#   ub -> upperbound of reals for both x and y variable
#   lb -> lowerbound of reals for both x and y variable 
#
# How to convert chromosome to two (bounded) real variables?
# The chromosome is splitted in half. The first half is used for x and the second one for y. 
# Idea: a segment XX' is described as all points between them: with all alpha belonging to {0,1} the segment is described as [alpha*X + (1-alpha)*X']
# the idea is to use that definition to approximate real numbers with that definition where alpha is a number between 0 and 1 converted from the half binary chromosome:
# Example: [0,1,1,0] = 1/4+1/8 = alpha 
# Increasing the number of bits of the chromosome implies that we have more real values represented between ub and lb.
# For peaks a good bound is (-3;+3)

class Peaks:
    def __init__(self,lbx,ubx,lby,uby):
        self.lbx = lbx
        self.ubx = ubx
        self.lby = lby
        self.uby = uby

    def bin_to_real_x(self,bin):
        num = 0
        for i in range(0,len(bin)):
            num += bin[i]*(2**-(i+1))
        return self.ubx*(num)+self.lbx*(1-num)
    
    def bin_to_real_y(self,bin):
        num = 0
        for i in range(0,len(bin)):
            num += bin[i]*(2**-(i+1))
        return self.uby*(num)+self.lby*(1-num)
    
    def get_value(self,chromosome):

        x = self.bin_to_real_x(bin = chromosome[:int(len(chromosome)/2)])
        y = self.bin_to_real_y(bin = chromosome[int(len(chromosome)/2):])
        e = math.e
        z = 3*((1-x)**2)*e**(-x**2-(y+1)**2) - 10*(x/5-x**3-y**5)*e**(-x**2-y**2)-(1/3)*e**(-(x+1)**2-y**2)
        return z

    def get_value_from_reals(self,x,y):
        e = math.e
        z = 3*((1-x)**2)*e**(-x**2-(y+1)**2) - 10*(x/5-x**3-y**5)*e**(-x**2-y**2)-(1/3)*e**(-(x+1)**2-y**2)
        return z
    

    # Plot the function and all the best chromosome for generation (number of generation as a marker). 
    def plot(self,history,generations):
        
        x_ = arange(self.lbx,self.ubx,0.25)
        y_ = arange(self.lby,self.uby,0.25)

        X,Y = meshgrid(x_, y_) # grid of point
        Z = self.get_value_from_reals(X,Y)


        counter = 0
        k = 1
        
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 1,1, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=CMAP,linewidth=0, antialiased=False, alpha = 0.5)
        
        for i in range(0,generations):
            xs = history[i][0]
            ys = history[i][1]
            zs = self.get_value_from_reals(xs, ys)
            ax.scatter(xs, ys, zs, marker="x",color = 'red',s = 50)

        plt.show()
   
    def plot_(self):
        
        x_ = arange(self.lbx,self.ubx,0.07)
        y_ = arange(self.lby,self.uby,0.07)

        X,Y = meshgrid(x_, y_) # grid of point
        Z = self.get_value_from_reals(X,Y)



        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 1,1, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = CMAP,linewidth=0, antialiased=False, alpha = 0.9)
        ax.contour(X,Y,Z,30,cmap = CMAP,offset=-7)
        plt.show()

    def plot_contour(self):
            x_ = arange(self.lbx,self.ubx,0.07)
            y_ = arange(self.lby,self.uby,0.07)

            X,Y = meshgrid(x_, y_) # grid of point
            Z = self.get_value_from_reals(X,Y)
            origin = "lower"
            fig1, ax1 = plt.subplots(constrained_layout=True)
            CS = ax1.contourf(X,Y,Z,30, cmap = CMAP, origin=origin)
            cbar = fig1.colorbar(CS)
            cbar.ax.set_ylabel('verbosity coefficient')
            # Add the contour line levels to the colorbar
            

            # to display value in the contour 
            #plt.clabel(contours, inline=1, fontsize=10)
            
            plt.show()

    def plot_contour_history(self,history,generations):
            x_ = arange(self.lbx,self.ubx,0.07)
            y_ = arange(self.lby,self.uby,0.07)

            X,Y = meshgrid(x_, y_) # grid of point
            Z = self.get_value_from_reals(X,Y)
            origin = "lower"
            fig1, ax1 = plt.subplots(constrained_layout=True)
            CS = ax1.contourf(X,Y,Z,30,cmap=CMAP, origin=origin)
            cbar = fig1.colorbar(CS)
            
            #cbar.ax.set_ylabel('verbosity coefficient')
            # Add the contour line levels to the colorbar
            
            x_ = []
            y_ = []
            for i in range(0,generations):
                xs = history[i][0]
                ys = history[i][1]
                x_.append(xs)
                y_.append(ys)
            
            # to display value in the contour 
            #plt.clabel(contours, inline=1, fontsize=10)
            plt.scatter(x_,y_,marker='x',c=COLOR)
            plt.show()        

# p = Peaks(-3,3,-3,3)
# p.plot_()
