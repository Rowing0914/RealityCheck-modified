import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import os
import time
import sys

def Parse_Config(Config):
    config_file = open(Config + ".txt")
    config_lines = config_file.readlines()
    lines = []
    for cl in config_lines:
        label, line = cl.strip().split(':')
        lines.append(line.strip())
    c00, c01, c02, c10, c11, c12, c20, c21, c22 = lines[0].split(',')
    real_dict = int(lines[1])
    x, y = lines[2].split(',')
    real_arrangement = [int(x.strip()),int(y.strip())]
    real_size = float(lines[3])
    real_space = float(lines[4])
    virtual_dict = int(lines[5])
    x, y = lines[6].split(',')
    virtual_arrangement = [int(x.strip()),int(y.strip())]
    virtual_size = float(lines[7])
    virtual_space = float(lines[8])
    camera = np.matrix([[float(c00.strip()), float(c01.strip()), float(c02.strip())],
                    [float(c10.strip()), float(c11.strip()), float(c12.strip())],
                    [float(c20.strip()), float(c21.strip()), float(c22.strip())]], dtype=np.float32)
    
    virtual_ardict = cv2.aruco.getPredefinedDictionary(virtual_dict)
    virtual_board = cv2.aruco.GridBoard_create(virtual_arrangement[0], virtual_arrangement[1], virtual_size, virtual_space, virtual_ardict)

    real_ardict = cv2.aruco.getPredefinedDictionary(real_dict)
    real_board = cv2.aruco.GridBoard_create(real_arrangement[0], real_arrangement[1], real_size, real_space, real_ardict)

    return camera, real_ardict, virtual_ardict, real_board, virtual_board
    


def GetBoardTransforms(imname, camera, real_ardict, virtual_ardict, real_board, virtual_board, ROTATE_IMAGE_90 = False):
    #load image
    img = cv2.imread(imname)

    if ROTATE_IMAGE_90:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)#COUNTER

    #distortion coeffs (assuming 0)
    distortion = []
    distort =  np.array(distortion, dtype = np.float32)

    virtual_corners, virtual_ids, rejected = cv2.aruco.detectMarkers(img,virtual_ardict)

    virtual_corners, virtual_ids, rejected, rid = cv2.aruco.refineDetectedMarkers(img,virtual_board,virtual_corners,virtual_ids,rejected, camera, distort)

    rvec = np.zeros(3)
    tvec = np.zeros(3)
    virtual_val , virtual_rvec, virtual_tvec = cv2.aruco.estimatePoseBoard(virtual_corners, virtual_ids, virtual_board, camera, distort, rvec, tvec) 
    
    real_corners, real_ids, rejected = cv2.aruco.detectMarkers(img,real_ardict)

    real_corners, real_ids, rejected, rid = cv2.aruco.refineDetectedMarkers(img,real_board,real_corners,real_ids,rejected, camera, distort)

    rvec = np.zeros(3)
    tvec = np.zeros(3)
    real_val , real_rvec, real_tvec = cv2.aruco.estimatePoseBoard(real_corners, real_ids, real_board, camera, distort, rvec, tvec) 

    return real_val, real_rvec, real_tvec, virtual_val, virtual_rvec, virtual_tvec

def Transform(p,R,t):
    #transform point p by Rotating by R(matrix) and translating by t(vector) 
    #helper function
    p = quaternion.rotate_vectors(R,p)
    return p + t

def TransformInverse(p,R,t):
    #transform point p by translating by -t(vector) and Rotating by R(matrix) inverse
    #helper function
    p -= t
    return quaternion.rotate_vectors(R.inverse(),p)

def FrameCheck(image, camera, real_ardict,virtual_ardict,real_board,virtual_board, rotate = False):
    print(image)

    r_val, r_rvec, r_tvec, v_val, v_rvec, v_tvec = GetBoardTransforms(image, camera, real_ardict, virtual_ardict, real_board, virtual_board, rotate)

    rtov = v_tvec - r_tvec

    distance = pow((rtov[0]*rtov[0] + rtov[1]*rtov[1] + rtov[2]*rtov[2]),0.5)
    
    return distance, v_tvec, r_tvec, v_rvec, r_rvec, v_val, r_val

def ParseAnnotation(fname):
    frames = []
    coords = []
    real_to_grid = [0.0,0.0]
    grid_size = []
    annotf = open(fname)
    annlines = annotf.readlines()
    for ann in range(len(annlines)):
        if (ann == 0):
            _, offset = annlines[ann].strip().split(':')
            x, y = offset.split(',')
            real_to_grid[0] = float(x.strip())
            real_to_grid[1] = float(y.strip())
        elif (ann == 1):
            _ , gs = annlines[ann].strip().split(':')
            grid_size = float(gs.strip())
        else:
            frameno, coord = annlines[ann].strip().split(':')
            frames.append(int(frameno.strip()))
            coordx, coordy = coord.split(',')
            coords.append([float(coordx.strip()),float(coordy.strip())])

    return frames, coords, grid_size, real_to_grid

def Annotated_distance(vcoords,grid_size, real_to_grid):
    dist = []
    for vc in vcoords:
        vc[0] = vc[0]*grid_size + real_to_grid[0]
        vc[1] = vc[1]*grid_size + real_to_grid[1]
        dist.append(math.pow(vc[0]*vc[0] + vc[1]*vc[1],0.5))
    return dist
    

def Write_results_to_file(Vod_Name, distances, v_tvecs, r_tvecs, v_rvecs, r_rvecs, v_vals, r_vals):
#file format is per frame:
#dist, [x,y,z], [x,y,z], [x,y,z], [x,y,z] v, r
#distance, virtual translation vector, real translation vector, virtual rotation vector, real rotation vector, virtual markers recognized, real markers recognised
    MarkFile = open(Vod_Name + "Output.txt", 'w')
    for i in range(len(v_tvecs)):
        MarkFile.write(str(distances[i]) + ", ")
        MarkFile.write(str(v_tvecs[i]) + ", ")
        MarkFile.write(str(r_tvecs[i]) + ", ")
        MarkFile.write(str(v_rvecs[i]) + ", ")
        MarkFile.write(str(r_rvecs[i]) + ", ")
        MarkFile.write(str(v_vals[i]) + ", ")
        MarkFile.write(str(r_vals[i]) + "\n")
    MarkFile.close()

def Graph_Trace(v_tvecs,r_tvecs):
    plt.title("Camera To Board Vectors")
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X', fontsize=35, labelpad = 35)
    ax.set_ylabel('Y', fontsize=35, labelpad = 35)
    ax.set_zlabel('Z', fontsize=35, labelpad = 20)
    xs = np.array([c[0] for c in v_tvecs])
    ys = np.array([c[1] for c in v_tvecs])
    zs = np.array([c[2] for c in v_tvecs])

    ax.plot3D(xs,ys,zs, 'blue')

    xs = np.array([c[0] for c in r_tvecs])
    ys = np.array([c[1] for c in r_tvecs])
    zs = np.array([c[2] for c in r_tvecs])

    ax.plot3D(xs,ys,zs, 'red')

    plt.tick_params(labelsize='20')

    plt.show()

def Graph_Distances(distances, frame_nos, ann_dist, graph_annotations = False):
    plt.ylabel("Distance Between Marker Boards (Meters)", fontsize = 25, labelpad = 10)
    plt.xlabel("Video Frame Number", fontsize = 30, labelpad = 10)

    data = np.array(distances)

    plt.scatter(range(len(data[:])),data[:], color='blue', label="RealityCheck")
    if (graph_annotations):
        plt.scatter(frame_nos, ann_dist, color = 'red', label="Ground Truth")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', fontsize = 20,
        ncol=2, mode="expand", borderaxespad=0.)
    plt.tick_params(labelsize='30')

    plt.show()


def Spatial_Consistency_Check(Vod_Dir, camera, real_ardict, virtual_ardict, real_board, virtual_board, rotate = False):

    distances = []
    v_tvecs = []
    r_tvecs = []
    v_rvecs = []
    r_rvecs = []
    r_vals = []
    v_vals = []

    im_list = os.listdir(Vod_Dir)
    im_list = sorted(im_list)

    for im in im_list[:]:
        dist, v_tvec, r_tvec, v_rvec, r_rvec, v_val, r_val = FrameCheck(Vod_Dir + "/" +  im, camera, real_ardict, virtual_ardict, real_board, virtual_board, rotate)
        distances.append(dist)
        v_tvecs.append(v_tvec)
        r_tvecs.append(r_tvec)
        v_rvecs.append(v_rvec)
        r_rvecs.append(r_rvec)
        v_vals.append(v_val)
        r_vals.append(r_val)

    return distances, v_tvecs, r_tvecs, v_rvecs, r_rvecs, v_vals, r_vals

def RealityCheck(Vod_Dir,Config, Annotate = False, Rotate = False):

    camera, real_ardict, virtual_ardict, real_board, virtual_board = Parse_Config(Config)

    start_time = time.time()
    distances, v_tvecs, r_tvecs, v_rvecs, r_rvecs, v_vals, r_vals = Spatial_Consistency_Check(Vod_Dir, camera, real_ardict, virtual_ardict, real_board, virtual_board, Rotate)
    print("--- %s seconds to execute---" % (time.time() - start_time))

    Graph_Trace(v_tvecs, r_tvecs)

    ann_dist = []
    frame_nos = []
    if (Annotate):
        frame_nos, virt_coords, grid_size, real_to_grid = ParseAnnotation(Vod_Dir + ".txt")
        ann_dist = Annotated_distance(virt_coords, grid_size, real_to_grid)

    Graph_Distances(distances, frame_nos, ann_dist, Annotate)

    Write_results_to_file(Vod_Name, distances, v_tvecs, r_tvecs, v_rvecs, r_rvecs, v_vals, r_vals)

    return 0

if __name__ == "__main__":
    Vod_Name = sys.argv[1]
    Config = sys.argv[2]
    annot = False
    rot = False
    if (len(sys.argv) > 3):
        annot = sys.argv[3] == 'True'
        if (len(sys.argv) > 4):
            rot = sys.argv[4] == 'True'

    RealityCheck(Vod_Name, Config, annot, rot)