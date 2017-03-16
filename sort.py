"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import cv2
import argparse
global seq,frame,imcounter
imcounter=0

if not os.path.exists('dets_and_trackers'):
    os.makedirs('dets_and_trackers')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument('--hum_width',dest='hum_width',type=int, help='resize feature to this width', default=64)
    parser.add_argument('--hum_height',dest='hum_height',type=int, help='resize feature to this height', default=128)
    parser.add_argument('--save_DA_images',dest='save_DA_images',type=bool,help='whether to save data association images or not', default=True)    
    parser.add_argument('--mot_impath',dest='mot_impath',help='path of mot data', default='/home/user/Datasets/mot_data/2DMOT2015/train/')    


    args = parser.parse_args()
    return args

def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

class BoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
  
    self.x=[]
    self.x[:4] = bbox
    
    self.time_since_update = 0
    self.id = BoxTracker.count
    BoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = bbox
    self.hits += 1
    self.hit_streak += 1
    self.x=bbox

  def predict(self,dets):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    
    return self.x

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.x
def show_what_to_associate(detections,trackers,img,args): #x1,y1,x2,y2
    global frame

    wtim=np.zeros([128,64,3])
    for d,det in enumerate(detections):
        crdet=crop(img,det)

        rcrdet=cv2.resize(crdet,(args.hum_width,args.hum_height), interpolation = cv2.INTER_NEAREST)
        
        wtim=np.concatenate((wtim,rcrdet),axis=1)  
    
    wtim=np.concatenate((wtim,np.ones([128,64,3])),axis=1)

    for t,trk in enumerate(trackers):
        
        crtrk=crop(img,trk)
        if min(crtrk.shape)==0:
            rcrtrk=np.ones([args.hum_height,args.hum_width,3])*170
        else:
            rcrtrk=cv2.resize(crtrk,(args.hum_width,args.hum_height), interpolation = cv2.INTER_NEAREST)

        wtim=np.concatenate((wtim,rcrtrk),axis=1)
    
    return wtim

def show_matched_pairs(detections,trackers,img,args,matches,unmatched_detections,unmatched_trackers,cost_matrix):#x1,y1,x2,y2
    global frame,imcounter
    wtim=show_what_to_associate(detections,trackers,img,args)

        
    lwtim=np.zeros([128,128,3])
    for m in matches:

        det=detections[m[0]]
        trk=trackers[m[1]]
        cost_text=cost_matrix[m[0],m[1]]        
    
        crdet=crop(img,det)
        crtrk=crop(img,trk)

        rcrdet=cv2.resize(crdet,(args.hum_width,args.hum_height), interpolation = cv2.INTER_NEAREST)

        if min(crtrk.shape)==0:
            rcrtrk=np.ones([args.hum_height,args.hum_width,3])*170
            cv2.putText(rcrtrk,'out_of_bounds', (5,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

        else:
            rcrtrk=cv2.resize(crtrk,(args.hum_width,args.hum_height), interpolation = cv2.INTER_NEAREST)

        dnt=np.concatenate((rcrdet,rcrtrk),axis=1)

        cv2.putText(dnt,str(float(cost_text)), (5,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        lwtim=np.concatenate((lwtim,dnt),axis=0)

    unmdetsim=np.zeros([128,128,3])
    tmp_unm_det=np.zeros([128,128,3])
    for m in unmatched_detections:

        det=detections[m]
      
        crdet=crop(img,det)

        rcrdet=cv2.resize(crdet,(args.hum_width,args.hum_height), interpolation = cv2.INTER_NEAREST)

        cv2.putText(rcrdet,'ud'+str(min(cost_matrix[int(m),:])), (5,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)

        tmp_unm_det=np.concatenate((rcrdet,unmdetsim),axis=1)       

    unmtrksim=np.zeros([128,128,3])
    tmp_unm_trk=np.zeros([128,128,3])
    for m in unmatched_trackers:

        trk=trackers[m]
    
        crtrk=crop(img,trk)

        if min(crtrk.shape)==0:
            rcrtrk=np.ones([args.hum_height,args.hum_width,3])*170
            cv2.putText(rcrtrk,'out_of_bounds', (5,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

        else:
            rcrtrk=cv2.resize(crtrk,(args.hum_width,args.hum_height), interpolation = cv2.INTER_NEAREST)
            

        cv2.putText(rcrtrk,'ut'+str( min(cost_matrix[:,int(m)])), (5,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)

        tmp_unm_trk=np.concatenate((rcrtrk,unmtrksim),axis=1)

    
    whole=np.zeros([wtim.shape[0]+lwtim.shape[0],wtim.shape[1]+lwtim.shape[1] , 3])

    whole[0:wtim.shape[0],0:wtim.shape[1],:]=wtim

    whole[0+128:lwtim.shape[0]+128,0+128:lwtim.shape[1]+128,:]=lwtim
   
    whole[0+128:128+tmp_unm_det.shape[0],0:tmp_unm_det.shape[1],:]=tmp_unm_det

    whole[0+128:128+tmp_unm_trk.shape[0],0+4*64:tmp_unm_trk.shape[1]+4*64,:]=tmp_unm_trk

    cv2.imwrite('dets_and_trackers/'+ str(frame)+'_'+str(imcounter)+'_dets_trackers_matches.jpg',(whole))
    imcounter=imcounter+1
    return
def fix_negative_coords(dets): #accepts x1,y1,x2,y2 np array Nx5
     
    h,w=dets.shape
    
    for i in xrange(0,h):
        if (dets[i][0])<0:
            dets[i][2]=dets[i][2]+dets[i][0]
            dets[i][0]=0
        if (dets[i][1])<0:
            dets[i][3]=dets[i][3]+dets[i][1]
            dets[i][1]=0

    return dets

def crop(imag,bx): #accepts x1,y1,x2,y2 numpy array
    
    tmbx=np.zeros([1,4])
    tmbx[0][0]=bx[0]
    tmbx[0][1]=bx[1]
    tmbx[0][2]=bx[2]
    tmbx[0][3]=bx[3]

    bx=fix_negative_coords(tmbx)
    x1=int(bx[0][0])
    y1=int(bx[0][1])
    x2=int(bx[0][2])
    y2=int(bx[0][3])    

    cropped = imag[y1:y2, x1:x2]     
    return cropped

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  global seq,frame
  
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  if args.save_DA_images==True:    
    impath=args.mot_impath+seq+'/img1/'+'%.6d'%frame+'.jpg'  
    img=cv2.imread(impath)
    show_matched_pairs(detections,trackers,img,args,matches,unmatched_detections,unmatched_trackers,iou_matrix)
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self,max_age=1,min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def update(self,dets):

    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))

    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(dets)
     
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
     
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:   
        trk = BoxTracker(dets[i,:]) 
        attrs = vars(trk)

        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()
    
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):

          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
    


if __name__ == '__main__':
  alltrk=[]
  # all train

  sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof','ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17','ADL-Rundle-6','ADL-Rundle-8','Venice-2']
  args = parse_args()
  display = args.display
  phase = 'train'
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32,3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure() 
  
  if not os.path.exists('output'):
    os.makedirs('output')
  
  for seq in sequences:
    mot_tracker = Sort() #create instance of the SORT tracker
    attrsm = vars(mot_tracker)
    #print ('class sort',' '.join("%s: %s\n" % item for item in attrsm.items()))

    seq_dets = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
    with open('output/%s.txt'%(seq),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:,0]==frame,2:7]
        dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          ax1 = fig.add_subplot(111, aspect='equal')
          fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq+' Tracked Targets')

        start_time = time.time()
        dets=dets.astype(int)
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        
        for d in trackers:
          alltrk.append(d)
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[5],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.uint32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
            ax1.set_adjustable('box-forced')

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()
   
  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  if(display):
    print("Note: to get real runtime results run without the option: --display")
