import numpy as np
import os
import math
import pickle

import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.markers import MarkerStyle
from scipy.ndimage import affine_transform
from skimage.util import montage
from scipy.stats.stats import pearsonr
from numpy.linalg import lstsq

from other import *

# quick plot of img and label pairs of the training data
# pngdir is just a quick hack for debugging
def display_img(input_gen, ax=None, delay=0.1, pngdir=None, vmax=100, pclose=False, mslice=False, ginput=None, order=3):

    # if pngdir:
    #     matplotlib.use('Agg')
    if pngdir:
        Path(pngdir).mkdir(parents=True,exist_ok=True)

    if ginput:
        pts = []
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(4,8))
    plt.tick_params(
        axis='both',
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False
    )
    if input_gen.do_plot is False:
        input_gen.do_plot = True

    ye = None
    for idx, (xi, yi) in enumerate(input_gen):
        dimsize = np.shape(np.squeeze(xi))
        xi = np.squeeze(xi) # remove dummy keras dimension
        D = 200
        L = 44.5 # mm
        if isinstance(yi,tuple): #prediction/error
            yp,ye = yi
            rx = np.deg2rad(yp[3]) # ie rotation about patient X (R/L)
            ry = np.deg2rad(yp[4]) # ie rotation about patient Y (A/P)
            Oz = yp[2]
            Ox = yp[1]
            Oy = yp[0]
            Oze = ye[2]
            Oxe = ye[1]
            Oye = ye[0]
            rxe = np.deg2rad(ye[3]) # ie rotation about patient X (R/L)
            rye = np.deg2rad(ye[4]) # ie rotation about patient Y (A/P)
        else:
            yi = np.squeeze(yi) # remove keras dummy dimension if any
            if len(yi)>3:
                rx = np.deg2rad(yi[3]) # ie rotation about patient X (R/L)
                ry = np.deg2rad(yi[4]) # ie rotation about patient Y (A/P)
            else:
                rx = 0
                ry = 0
            Oz = yi[2]
            Ox = yi[1]
            Oy = yi[0]
            yp = yi
            ye = None
        # resample volume to ua centre line
        R = get_rotmat(ry,0,rx) # in dimension order, patient X (R/L) is 3rd dim, patient Y (A/P) is 1st dim
        I = get_rotmat(0,0,0)
        # affine transform to shift centre of rotation to UA centre. doesn't include the through-plane offset yet
        if 1:
            T = get_affine(R,yp[0],yp[1],0)
            xi_resamp = affine_transform(xi,T[:3,:3],offset=T[:3,3],order=order)
            # then additionally shift to centre of the crop
            recentre_offset = np.array([yp[0]-dimsize[0]//2,yp[1]-dimsize[1]//2,0])
            xi_resamp = affine_transform(xi_resamp,I,offset=recentre_offset,order=order)
            xi_plot = xi_resamp[:,:,round(yp[2])]
            # multi-slice montage
            if mslice:
                xi_plotm = montage(np.transpose(xi[:,:,range(dimsize[2]//2-3,dimsize[2]//2+4)],(2,0,1)),grid_shape=(7,1))
                zi_resamp = range(max(0,round(Oz)-3),min(dimsize[2],round(Oz)+4))            
                xi_resamp_plotm = montage(np.transpose(xi_resamp[:,:,zi_resamp],(2,0,1)),grid_shape=(7,1))
                if len(zi_resamp) < 7:
                    xi_resamp_plotm = np.roll(xi_resamp_plotm,dimsize[0]*(7-len(zi_resamp)),axis=0)
                xi_montage = np.concatenate((xi_plotm,xi_resamp_plotm),axis=1)
        else: # through-plane offset
            T = get_affine(R,yp[0],yp[1],yp[2])
            xi_resamp = affine_transform(xi,T[:3,:3],offset=T[:3,3],order=order)
            recentre_offset = np.array([yp[0]-dimsize[0]//2,yp[1]-dimsize[1]//2,yp[2]-dimsize[2]//2])
            xi_resamp = affine_transform(xi_resamp,I,offset=recentre_offset,order=order)
            xi_plot = xi_resamp[:,:,dimsize[2]//2]
            # multi-slice montage
            if mslice:
                xi_plotm = montage(np.transpose(xi[:,:,range(dimsize[2]//2-3,dimsize[2]//2+4)],(2,0,1)),grid_shape=(7,1))
                zi_resamp = range(max(0,round(Oz)-3),min(dimsize[2],round(Oz)+4))            
                xi_resamp_plotm = montage(np.transpose(xi_resamp[:,:,zi_resamp],(2,0,1)),grid_shape=(7,1))
                if len(zi_resamp) < 7:
                    xi_resamp_plotm = np.roll(xi_resamp_plotm,dimsize[0]*(7-len(zi_resamp)),axis=0)
                xi_montage = np.concatenate((xi_plotm,xi_resamp_plotm),axis=1)

        # debugging plots
        if False:
            plt.figure(10,figsize=(8,10))
            plt.subplot(1,1,1)
            if not vmax:
                vmax = np.max(xi_plotm) / 2
            plt.imshow(xi_montage,vmax=vmax)
            plt.text(5,-5,'rx={:.2}, ry={:.2}'.format(yi[3],yi[4]))
            plt.text(5,-15,'Ox={:.2f}, Oy={:.2f}, Oz={:.2f}'.format(yi[0],yi[1],yi[2]))
            plt.plot(yi[1],yi[0] + (int(Oz)-4)*dimsize[0],'r+')
            plt.plot(dimsize[1]//2-1 + dimsize[1],dimsize[0]//2-1 + 3*dimsize[0],'r+')
            # plt.show()
            a=1

        if mslice: # substitute mult-slice montage
            xi_plot = xi_montage

        # image data
        if not vmax:
            vmax = np.max(xi_plot) / 2
        if idx == 0:
            im_1 = ax.imshow(np.squeeze(xi_plot), vmin=0, vmax=vmax, origin='upper')
            plt.xlim(0,np.shape(xi_plot)[1])
            plt.ylim(0,np.shape(xi_plot)[0])
            ax.invert_yaxis() # maintina y origin at top for coords convention
            ax.axis('off')
        else:
            im_1.set_data(np.squeeze(xi_plot))
        if False:
            plt.title('idx = {}'.format(idx))
            plt.plot(64,32,'r+')

        # overlay
        if 1:
            if idx > 0:
                l_cross1.pop(0).remove()
                if mslice:
                    l_cross2.pop(0).remove()
            if mslice:
                l_cross1 = ax.plot(yp[1],yp[0] + (round(Oz)-dimsize[2]//2+3)+3*dimsize[0],'r+')
                l_cross2 = ax.plot(dimsize[1]//2-1 + dimsize[1],dimsize[0]//2-1 + 3*dimsize[0],'r+')
            else:
                if 0:
                    l_cross1 = ax.plot(yp[1],yp[0],'r+')
                else:
                    l_cross1 = ax.plot(dimsize[1]//2-0,dimsize[0]//2-0,'r+')


        # annotation
        if idx > 0:
            l_1.set_visible(False)
            m_1.set_visible(False)
            if ye is not None:
                m_2.set_visible(False)
                m_3.set_visible(False)
                l_2.set_visible(False)

        if ye is None:
            if 1:
                l_1 = ax.text(5,-5,'rx,ry = {:.1f},{:.1f}'.format(np.rad2deg(rx),np.rad2deg(ry)),{'fontsize':8})
                m_1 = ax.text(5,-30,'px,py,pz = {:.1f},{:.1f},{:.1f}'.format(Ox,Oy,Oz),{'fontsize':8})
            else:
                l_1 = ax.text(5,-5,'rx,ry = {:.1f},{:.1f}'.format(np.rad2deg(rx),np.rad2deg(ry)),{'fontsize':8})
                m_1 = ax.text(5,-9,'px,py,pz = {:.1f},{:.1f},{:.1f}'.format(Ox,Oy,Oz),{'fontsize':8})
        else:
            if 1:
                l_1 = ax.text(5, -5,'{:<10} = {:6.1f} ({:4.1f})'.format('phi',ry*180/np.pi,rye*180/np.pi),{'fontsize':8},family='monospace')
                l_2 = ax.text(5, -14,'{:<10} = {:6.1f} ({:4.1f})'.format('theta',rx*180/np.pi,rxe*180/np.pi),{'fontsize':8},family='monospace')
                m_1 = ax.text(5,-23,'{:<10} = {:6.1f} ({:4.1f})'.format('pz',Oz,Oze),{'fontsize':8},family='monospace')
                m_2 = ax.text(5,-32,'{:<10} = {:6.1f} ({:4.1f})'.format('py',Oy,Oye),{'fontsize':8},family='monospace')
                m_3 = ax.text(5,-41,'{:<10} = {:6.1f} ({:4.1f})'.format('px (err)',Ox,Oxe),{'fontsize':8},family='monospace')
            else:
                l_1 = ax.text(5, -5,'{:<10} = {:6.1f} ({:4.1f})'.format('phi',ry*180/np.pi,rye*180/np.pi),{'fontsize':8},family='monospace')
                l_2 = ax.text(5, -14,'{:<10} = {:6.1f} ({:4.1f})'.format('theta',rx*180/np.pi,rxe*180/np.pi),{'fontsize':8},family='monospace')
                m_1 = ax.text(5,-23,'{:<10} = {:6.1f} ({:4.1f})'.format('pz',Oz,Oze),{'fontsize':8},family='monospace')
                m_2 = ax.text(5,-32,'{:<10} = {:6.1f} ({:4.1f})'.format('py',Oy,Oye),{'fontsize':8},family='monospace')
                m_3 = ax.text(5,-41,'{:<10} = {:6.1f} ({:4.1f})'.format('px (err)',Ox,Oxe),{'fontsize':8},family='monospace')

            # l_1b = ax.text(5,dimsize[0]+4,'error = {:.1f}'.format(ye[2]),{'fontsize':8})

        if 1:
            if idx == 0:
                plt.pause(1*delay)
            else:
                plt.pause(delay)
            if ginput:
                pt = plt.ginput(ginput)
                pts.append( pt )

        # output for further debugging
        if pngdir:
            plt.savefig(os.path.join(pngdir,'I{:05d}.png'.format(idx)), bbox_inches='tight')

    input_gen.do_plot = False

    if pclose:
        plt.close()
    if ginput:
        return pts
    else:
        return ax


# training results plots
# 4 panel comp plot 5dof
def plot_res(model,res,xvals,avals,trainhistory,rmse_deg,rmse_mm,outliers,outliers_mm,errvals,output_size,deglim,pxlim,pngdir,datagen):

    ll = ['y','x','z','theta','phi']

    fig1 = plt.subplots(1,4,figsize=(12,4))
    ax = plt.subplot(1,4,1)
    loss = trainhistory['history']['loss']
    val_loss = trainhistory['history']['val_loss']
    comb_loss = np.concatenate((loss,val_loss))
    losslim = (np.power(10.,int(math.floor(math.log10(abs(np.min(comb_loss)))))) , 
                np.power(10,int(math.ceil(math.log10(abs(np.max(comb_loss)))))))
    epochs = range(0,len(loss))
    ax.plot(epochs,loss,'r-',label='train')
    ax.plot(epochs,val_loss,'g-',label='validate')
    plt.ylim(losslim)
    plt.xlim(0,len(loss))
    plt.xlim(0,100)
    plt.yscale('log')
    ax.set_aspect((len(epochs))/( np.log10(losslim[1]/losslim[0]) ))
    plt.title('training,validation loss',{'fontsize':10})
    plt.xlabel('epoch')
    plt.legend(('train','validate'))
    plt.xticks(range(0,len(epochs)+10,100))

    # prediction error histo
    nh = np.empty(output_size)
    hl = np.empty(output_size)
    bins = np.arange(-5,5,.5)
    for i in range(output_size):
        nh[i] = np.max(np.histogram(errvals[:,i],bins=bins,density=True)[0])
    histmax = np.max(nh)*1.1
    histlim = (0,output_size*histmax)
    ax2 = plt.subplot(1,4,2)
    for i,b in enumerate([3, 4, 2, 1, 0]):
        ax2.hist(errvals[:,i],bins=bins,alpha=1.0,density=True,bottom=b*histmax,label=ll[i])
    plt.xlabel('prediction error (deg,mm)')
    plt.xlim(deglim)
    plt.ylim(histlim)
    ax2.set_aspect(np.diff(deglim)/np.diff(histlim))
    plt.xticks(range(deglim[0]+1,deglim[1]+1,2))
    plt.yticks(np.arange(0,output_size,1)*histmax,labels=(('0')*output_size))
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0,2,3,4]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='lower right',bbox_to_anchor=(.25,.85),fontsize=6,framealpha=1)

    # prediction error plots versus angle
    fulldeglim=(-10,30)
    ax3 = plt.subplot(1,4,3)
    for i in range(3,output_size):
        plt.plot(avals[:,i],errvals[:,i],'+',color=ax2.axes.containers[i].patches[0].get_facecolor())
    plt.xlabel('angle (deg)')
    plt.title('prediction error (deg)',{'fontsize':10})
    plt.xlim(fulldeglim)
    plt.ylim(deglim)
    plt.xticks(range(0,fulldeglim[1]+1,fulldeglim[1]//3))
    ax3.text(fulldeglim[0]+3,deglim[1]-1,'rmse_deg (outliers) = {:.2f} ({:d})'.format(rmse_deg,outliers),fontsize=8)
    ax3.text(fulldeglim[0]+3,deglim[1]-2,'rmse_mm = {:.2f} ({:d})'.format(rmse_mm,outliers_mm),fontsize=8)
    ax3.set_aspect(np.diff(fulldeglim)/np.diff(deglim))

    # plt.show()

    # movie
    ax = plt.subplot(1,4,4)
    # display_img(datagen.displayall_gen(nth=1,res=res), delay=0.3,ax=ax, vmax=0, pngdir=pngdir)
    display_img(datagen, delay=0.3,ax=ax, vmax=0, pngdir=pngdir)
    # display_img(datagen, delay=0.3,ax=ax, vmax=0, pngdir=pngdir,mslice=False)
    if pngdir is not None:
        os.system('convert -delay 100 {}/I*png {}/{}.mp4'.format(pngdir,pngdir,model))


# training results plots
# 4 panel comp plot 3dof
def plot_res_3dof(model,res,xvals,avals,trainhistory,rmse_deg,rmse_mm,outliers,outliers_mm,errvals,output_size,deglim,pxlim,pngdir,validation_datagen):

    ll = ['y','x','z']

    fig1 = plt.subplots(1,4,figsize=(12,4))
    ax = plt.subplot(1,2,1)
    loss = trainhistory['history']['loss']
    val_loss = trainhistory['history']['val_loss']
    comb_loss = np.concatenate((loss,val_loss))
    losslim = (np.power(10.,int(math.floor(math.log10(abs(np.min(comb_loss)))))) , 
                np.power(10,int(math.ceil(math.log10(abs(np.max(comb_loss)))))))
    epochs = range(0,len(loss))
    ax.plot(epochs,loss,'r-',label='train')
    ax.plot(epochs,val_loss,'g-',label='validate')
    plt.ylim(losslim)
    plt.xlim(0,len(loss))
    plt.xlim(0,100)
    plt.yscale('log')
    ax.set_aspect((len(epochs))/( np.log10(losslim[1]/losslim[0]) ))
    # ax.set_aspect((100)/( np.log10(losslim[1]/losslim[0]) ))
    plt.title('training,validation loss',{'fontsize':10})
    plt.xlabel('epoch')
    plt.legend(('train','validate'))
    plt.xticks(range(0,len(epochs)+10,100))

    # prediction error histo
    nh = np.empty(output_size)
    hl = np.empty(output_size)
    bins = np.arange(pxlim[0],pxlim[1],.5)
    for i in range(output_size):
        nh[i] = np.max(np.histogram(errvals[:,i],bins=bins,density=True)[0])
    histmax = np.max(nh)*1.1
    histlim = (0,output_size*histmax)
    ax2 = plt.subplot(1,2,2)
    for i,b in enumerate([1, 2, 0]):
        ax2.hist(errvals[:,i],bins=bins,alpha=1.0,density=True,bottom=b*histmax,label=ll[i])
        if i==1:
            ax2.text(pxlim[0],(b+.8)*histmax,'rmse (outliers)= {:.1f} ({:d})'.format(rmse_mm[i],outliers_mm[i]),fontsize=8)
        else:
            ax2.text(pxlim[0],(b+.8)*histmax,'rmse = {:.1f} ({:d})'.format(rmse_mm[i],outliers_mm[i]),fontsize=8)
    plt.xlabel('prediction error (mm)')
    plt.xlim(pxlim)
    plt.ylim(histlim)
    ax2.set_aspect(np.diff(pxlim)/np.diff(histlim))
    plt.xticks(range(pxlim[0]+1,pxlim[1]+1,2))
    plt.yticks(np.arange(0,output_size,1)*histmax,labels=(('0')*output_size))
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1.05,1.05),fontsize=6,framealpha=1)

    # plt.show()

    if pngdir is not None:
        if not os.path.exists(pngdir):
            os.mkdir(pngdir)
        plt.savefig(os.path.join(pngdir,'rmse_mm.png'),bbox_inches='tight')



# 1 panel resampled plot
def fit_res(model,res,xvals,avals,trainhistory,errvals,output_size,pngdir,validation_datagen):

    ll = ['py (mm)','px (mm)','z','theta (deg)','phi']
    epochs = range(0,len(trainhistory['history']['loss']))
    deg_lim=(-5,5)

    ptsfile = os.path.join(pngdir,'UApts.pickle')

    # record fiducial and 2nd point on axis of UA
    if os.path.exists(ptsfile):
        with open(ptsfile,'rb') as fp:
            pts = pickle.load(fp)
    else:
        fig1 = plt.subplots(1,1,figsize=(12,6))
        ax = plt.subplot(1,1,1)
        pts = display_img(validation_datagen.displayall_gen(nth=1,res=res), delay=1.0,ax=ax, vmax=0, ginput=2)
        with open(ptsfile,'wb') as fp:
            pickle.dump(pts,fp)

    xy=[]
    xy2=[]
    for p in range(len(pts)):
        xy.append(pts[p][0][::-1])
        xy2.append(pts[p][1][::-1])
    xy = np.array(xy)
    xy2 = np.array(xy2)

    apparent_errvals = np.zeros(np.shape(errvals))
    apparent_errvals[:,0:2] = xy - np.mean(xy,axis=0)
    thta = np.rad2deg(np.arctan2((xy2[:,1]-xy[:,1]),(xy2[:,0]-xy[:,0])))
    apparent_errvals[:,3] = thta - np.mean(thta)

    fig2 = plt.subplots(1,3,figsize=(9,3))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for idx,e in enumerate([1,0,3]):

        x = errvals[:,e]
        y = apparent_errvals[:,e]
        r,_ = pearsonr(x,y)    
        A = np.vstack([x, np.ones(len(x))]).T
        ab,_,_,_=lstsq(A, y)
        xfit = np.arange(deg_lim[0],deg_lim[1]+1,1)
        yfit = ab[0]*xfit + ab[1]

        ax = plt.subplot(1,3,idx+1)
        plt.plot(x,y,'+',color=colors[e])
        plt.xlim(deg_lim)
        plt.ylim(deg_lim)
        plt.xticks(range(-4,6,2))
        plt.plot(xfit,yfit,'k')
        plt.text(-4,4,ll[e])
        plt.text(-4,3,'r = {:.2f}'.format(r))
        ax.set_aspect(1)

        if idx == 0:
            plt.xlabel('delta from label')
            plt.ylabel('delta measured')
            plt.text(6,6,'Prediction errors: Labels versus Measured')
        elif idx>0:
            plt.tick_params(axis='x',        bottom=True,        labelbottom=False)
            plt.tick_params(axis='y',        left=True,        labelleft=False)


    # plt.show()
    plt.savefig(os.path.join(pngdir,'UApts.png'), bbox_inches='tight')
    a=1
