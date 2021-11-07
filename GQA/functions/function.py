import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from numpy import *
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import os

e = math.e
pi = math.pi

dirname = os.path.dirname(__file__)


class Function:
    def __init__(self,file):
        file1 = open(os.path.join(dirname,"input_functions",file), 'r')
        Lines_ = file1.readlines()
        Lines = []

        for l in Lines_:
            Lines.append(l.replace("\n", ""))

        self.function = Lines[0]
        self.lbx = float(Lines[1])
        self.ubx = float(Lines[2])
        self.lby = float(Lines[3])
        self.uby = float(Lines[4])

        
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
        z = eval(self.function)
        return z

    def get_value_from_reals(self,x,y):
        z = eval(self.function)
        return z
        
    

    # Plot the function and all the best chromosome for generation (number of generation as a marker). 
    def plot(self,history,generations):
        
        x_ = arange(self.lbx,self.ubx,0.09)
        y_ = arange(self.lby,self.uby,0.09)

        X,Y = meshgrid(x_, y_) # grid of point
        Z = self.get_value_from_reals(X,Y)


        counter = 0
        k = 1
        
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 1,1, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.RdBu,linewidth=0, antialiased=False, alpha = 0.45)
        
        for i in range(0,generations):
            xs = history[i][0]
            ys = history[i][1]
            zs = self.get_value_from_reals(xs, ys)
            ax.scatter(xs, ys, zs, marker="x",color = 'red',s = 50)

        plt.show()
            
    def plot_(self):
        rx = (self.ubx - self.lbx)/100
        ry = (self.uby - self.lby)/100
        x_ = arange(self.lbx,self.ubx,rx)
        y_ = arange(self.lby,self.uby,ry)

        X,Y = meshgrid(x_, y_) # grid of point
        Z = self.get_value_from_reals(X,Y)
        
        fig = plt.figure(figsize=plt.figaspect(0.8))

        ax = fig.add_subplot(1, 1,1, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,linewidth=0, cmap = cm.inferno,antialiased=False, alpha = 0.3)
        plt.show()

    def plot_contour(self):
            rx = (self.ubx - self.lbx)/100
            ry = (self.uby - self.lby)/100
            x_ = arange(self.lbx,self.ubx,rx)
            y_ = arange(self.lby,self.uby,ry)

            X,Y = meshgrid(x_, y_) # grid of point
            Z = self.get_value_from_reals(X,Y)
            origin = "lower"
            fig1, ax1 = plt.subplots(constrained_layout=True)
            CS = ax1.contourf(X,Y,Z,300,cmap = cm.inferno,origin=origin)
            cbar = fig1.colorbar(CS)
            cbar.ax.set_ylabel('verbosity coefficient')
            # Add the contour line levels to the colorbar
            

            # to display value in the contour 
            #plt.clabel(contours, inline=1, fontsize=10)
            
            plt.show()

    def plot_contour_history(self,history,generations):
            
            rx = (self.ubx - self.lbx)/100
            ry = (self.uby - self.lby)/100
            x_ = arange(self.lbx,self.ubx,rx)
            y_ = arange(self.lby,self.uby,ry)

            X,Y = meshgrid(x_, y_) # grid of point
            Z = self.get_value_from_reals(X,Y)
            origin = "lower"
            fig1, ax1 = plt.subplots(constrained_layout=True)
            CS = ax1.contourf(X,Y,Z,300,cmap = cm.inferno,origin=origin)
            cbar = fig1.colorbar(CS)
            cbar.ax.set_ylabel('verbosity coefficient')
            # Add the contour line levels to the colorbar
            

            # to display value in the contour 
            #plt.clabel(contours, inline=1, fontsize=10)
            for i in range(0,generations):
                xs = history[i][0]
                ys = history[i][1]
                
                ax1.scatter(xs, ys, marker="x",color = 'red',s = 50)
            
            plt.show()


# f = Function("func1.txt")
# print(f.get_value_from_reals(3,2))
# f.plot_()
# f.plot_contour()