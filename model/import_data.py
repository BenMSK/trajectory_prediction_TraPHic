# -*- coding: utf-8 -*-
import re
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import sys
# np.set_printoptions(threshold=sys.maxsize)


def import_data(file_dir, homography_dir, out_dir, class_type):
    transform(file_dir, homography_dir, out_dir, class_type)

def merge(file_names, output_dir, dtype, threadid, class_type):
    '''
    file_names: .npy file's lists
    output_dir: txt set list, without .txt
    결국, merge가 하는 일은 dsId.txt들을 하나의 .npy로 정리한 것.
    
    raw dataset:
    dsId1.txt, dsId2.txt, dsId3.txt, ..., dsIdN.txt

    final dataset for NN:
    Set#-traj.npy: 
    [ [ [dsId1, vehId(1), frame1, x, y, ----- i's features -----],
        [dsId1, vehId(2), frame1, x, y, ----- i's features -----],
        [dsId1, vehId(3), frame1, x, y, ----- i's features -----],
                                ... ,
        [dsId1, vehId(f1), frame1, x, y, ----- i's features -----],
        [dsId1, vehId(1), frame2, x, y, ----- i's features -----],
        [dsId1, vehId(2), frame2, x, y, ----- i's features -----],
                                ... ,
        [dsId1, vehId(fM_1), frameM_1, x, y, ----- i's features -----],
        

        [dsId2, vehId(1), frame1, x, y, ----- i's features -----],
                                ... ,
        [dsId2, vehId(fM_2), frameM_2, x, y, ----- i's features -----],


                                ... ,
        [dsIdN, vehId(i), frame, x, y, ----- i's features -----] ] ]

    Set#-track.npy:
      [ {
            dsId1: {
                    vehId(1): [frame, x, y]'s history,
                    vehId(2): [frame, x, y]'s history,
                                    ... ,
                    vehId(p1): [frame, x, y]'s history
                    }

            dsId2:  {
                    vehId(1): [frame, x, y]'s history,
                    vehId(2): [frame, x, y]'s history,
                                    ... ,
                    vehId(p2): [frame, x, y]'s history
                    }
                ...
            
            dsIdN:  {
                    vehId(1): [frame, x, y]'s history,
                    vehId(2): [frame, x, y]'s history,
                                    ... ,
                    vehId(pN): [frame, x, y]'s history
                    }
        } ]
    '''
    output_dir = output_dir + '/{}'
    traj = np.array([])

    track = defaultdict(dict)

    i = 0
    sz = len(file_names)
    for f in file_names:#get .npy file
        print("Start merging {}/{} in {} in thread {}...".format(i, sz, dtype, threadid))
        i += 1
        # print("Reading dataset {}...".format(d))
        npy_path = f
        print('reading... ', npy_path)
        data = np.load(npy_path, allow_pickle=True)# get formated .npy file

        # print('data', data.shape)
        # constructing train, val and testset for trajectory
        data0 = data[0]#traj data for a specific class
        data2 = data[2]#traj data for all
        
        traj_id = np.unique(data2[:,1])# get all object_id from dset 'd' if you make dataset for specific agent type
        
        if len(data0)==0:
            continue
        d = int(data0[0, 0])#dataset id
        

        if traj.size == 0:
            traj = data0
        else:
            '''# traj는 모든 #.npy의 traj를 합친 파일.'''
            traj = np.concatenate((traj, data0), axis=0)

        # constructing train, val and testset for tracks
        data1 = data[1]#track dataset from 'transform' function
        for ids in traj_id:#(t, x, y)
            '''# track은 모든 .npy의 track을 저장한 파일.'''
            track[d][ids] = data1[ids]#literally, each object's trajectory

        # print("Dataset {} finsihed.".format(d))
    
    if not os.path.exists(output_dir.format(dtype)):
        os.makedirs(output_dir.format(dtype))

    # data for sgan
    # sgan_name = "{}/{}Set{}.txt".format(dtype, dtype, str(threadid))
    # f = open(output_dir.format(sgan_name), 'w')
    # for line in traj:
    #     # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
    #     f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    # f.close()

    if class_type == 'vehicle':
        npy_name = "{}/{}Set{}-traj-v.npy".format(dtype, dtype, str(threadid))
    elif class_type == 'bike/motor':
        npy_name = "{}/{}Set{}-traj-b.npy".format(dtype, dtype, str(threadid))
    elif class_type == 'human':
        npy_name = "{}/{}Set{}-traj-h.npy".format(dtype, dtype, str(threadid))
    else:
        npy_name = "{}/{}Set{}-traj.npy".format(dtype, dtype, str(threadid))

    name = npy_name
    np.save(output_dir.format(name), np.array([traj]))
    name = "{}/{}Set{}-track.npy".format(dtype, dtype, str(threadid))
    np.save(output_dir.format(name), np.array([track]))
    
    print("{} file in thread {} is saved and ready.".format(dtype, threadid))

    return len(traj)

def merge_n_split(file_names, output_dir):

    output_dir = output_dir + '/{}'
    traj_train = np.array([])
    traj_val = np.array([])
    traj_test = np.array([])

    track_train = defaultdict(dict)
    track_val = defaultdict(dict)
    track_test = defaultdict(dict)

    print("Start spliting data...")

    for f in file_names:
        # print("Reading dataset {}...".format(d))
        npy_path = f
        # print(npy_path)
        data = np.load(npy_path, allow_pickle=True)

        # constructing train, val and testset for trajectory
        traj = data[0]
        traj_id = np.unique(traj[:,1])
        d = int(traj[0,0])

        # split the dataset using vehicle id
        traj_id, test_id = train_test_split(traj_id, test_size=0.2, random_state=0)
        train_id, val_id = train_test_split(traj_id, test_size=0.125, random_state=0)

        # get trajectory with vehicle id
        train = np.array(traj[ [(s in train_id) for s in traj[:,1]] ])
        if traj_train.size == 0:
            traj_train = train
        else:
            traj_train = np.concatenate((traj_train, train), axis=0)

        val = np.array(traj[ [(s in val_id) for s in traj[:,1]] ])
        if traj_val.size == 0:
            traj_val = val
        else:
            traj_val = np.concatenate((traj_val, val), axis=0)

        test = np.array(traj[ [(s in test_id) for s in traj[:,1]] ])
        if traj_test.size == 0:
            traj_test = test
        else:
            traj_test = np.concatenate((traj_test, test), axis=0)


        # constructing train, val and testset for tracks
        track = data[1]
        for i in train_id:
            track_train[d][i] = track[i]

        for i in val_id:
            track_val[d][i] = track[i]

        for i in test_id:
            track_test[d][i] = track[i]     

        print("Dataset {} finsihed.".format(d))



    if not os.path.exists(output_dir.format("train")):
        os.makedirs(output_dir.format("train"))

    f = open(output_dir.format("train/TrainSet.txt"), 'w')
    for line in traj_train:
        # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
        f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    f.close()

    if not os.path.exists(output_dir.format("val")):
        os.makedirs(output_dir.format("val"))

    f = open(output_dir.format("val/ValSet.txt"), 'w')
    for line in traj_val:
        # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
        f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    f.close()

    if not os.path.exists(output_dir.format("test")):
        os.makedirs(output_dir.format("test"))    

    f = open(output_dir.format("test/TestSet.txt"), 'w')
    for line in traj_test:
        # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
        f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    f.close()

    np.save(output_dir.format("TrainSet.npy"), np.array([traj_train, track_train]))
    np.save(output_dir.format("ValSet.npy"), np.array([traj_val, track_val])) 
    np.save(output_dir.format("TestSet.npy"), np.array([traj_test, track_test]))
    print("Training file saved and ready.")

    # print(traj_train)
    # print(traj_val)
    # print(traj_test)
    # print(len(traj_train))
    # print(len(traj_val))
    # print(len(traj_test))
    # for d in 
    # print(len(traj_train))
    # print(len(traj_val))
    # print(len(traj_test))
    # print(traj_val)


def filter_edge_cases(traj, track): 
    size = np.shape(traj)[0]
    idx = np.zeros((size, 1))

    for k in range(size):
        t = traj[k,2]

        if np.shape(track[traj[k,1]])[1] > 30 and track[traj[k,1]][0, 30] < t and track[traj[k,1]][0,-1] > t+1:
            idx[k] = 1       

    return traj[np.where(idx == 1)[0],:]




def px_to_ft(traj, homography_dir):

    m_to_ft = 3.28084
    h = np.loadtxt(homography_dir, delimiter=' ')
    c_x = 1280/2
    c_y = 720/2

    traj[:,3] = traj[:,3] - c_x
    traj[:,4] = traj[:,4] - c_y
    traj[:,3:5] = multiply_homography(h, traj[:,3:5]) * m_to_ft
    print("Finish converting pixel to feet")
    return traj

def multiply_homography(h, pt_in):
    # a = np.transpose(pt_in)
    # b = np.ones((1, np.shape(pt_in)[0]))
    # print(np.concatenate((a,b)))
    pt = np.matmul(h, np.concatenate((np.transpose(pt_in), np.ones((1, np.shape(pt_in)[0])))))
    pt = np.transpose(pt[0:2,:])
    # print(pt)
    return pt

def transform(file_dir, homography_dir, out_dir, class_type):
    #transform(formated_txt_file_dir, None, formated_npy file_dir)
    read = np.loadtxt(file_dir, delimiter=',')
    traj = np.zeros((np.shape(read)[0], 47))
    traj[:,:5] = read[:,:5]
    
    # uniq_id = np.unique(traj)
    ############ preprocess ##################
    read[read[:,5]==2,5]=1# only vehicle
    read_v = read[read[:,5]==1]
    traj_v = np.zeros((np.shape(read_v)[0], 47))
    traj_v[:,:5] = read_v[:,:5]

    read_b = read[read[:,5]==4]# only bike/motorcycle
    traj_b = np.zeros((np.shape(read_b)[0], 47))
    traj_b[:,:5] = read_b[:,:5]
    
    read_h = read[read[:,5]==3]# only human
    traj_h = np.zeros((np.shape(read_h)[0], 47))
    traj_h[:,:5] = read_h[:,:5]
    
    traj_class = None
    if class_type == 'vehicle':
        traj_class = traj_v
    elif class_type == 'bike/motor':
        traj_class = traj_b
    elif class_type == 'human':
        traj_class = traj_h
    else:# for all
        traj_class = traj
    

    # for k in range(np.shape(traj)[0]):#Ben: each row index(k)
    for k in range(np.shape(traj_class)[0]):#Ben: each row index(k)
        # print("Progress: {}/{} ...".format((k+1), np.shape(traj)[0]))
        dsid = traj_class[k][0]#traj_v
        vehid = traj_class[k][1]#traj_v
        time = traj_class[k][2]#traj_v
        
        #Get observed vehicle's trajectory
        vehtraj = traj_class[traj_class[:,1] == vehid]
        
        #Get all rows which are in the same frame
        frameEgo = traj[traj[:,2] == time]
        
        ''' Get Features '''
        if frameEgo.size != 0:#In one frame, there are a lot of dynamic obstacles
            dx = np.zeros(np.shape(frameEgo)[0])
            dy = np.zeros(np.shape(frameEgo)[0])
            vid = np.zeros(np.shape(frameEgo)[0])
            
            # agent_i를 기준으로 같은 frame 내의 object들과의 dx, dy 구하기.
            for l in range(np.shape(frameEgo)[0]):
                dx[l] = frameEgo[l][3] - traj_class[k][3]#traj_v
                dy[l] = frameEgo[l][4] - traj_class[k][4]
                vid[l] = frameEgo[l][1]
            dist = dx*dx + dy*dy# Get the distance between others from that vehicle;
            
            lim = 39# maximum 39 dynamic obstacles only

            if len(dist) > lim:
                idx = np.argsort(dist)
                # print(idx)
                dx = np.array([dx[i] for i in idx[:lim]])
                dy = np.array([dy[i] for i in idx[:lim]])
                vid = np.array([vid[i] for i in idx[:lim]])

            # left
            xl = dx[dx < 0]
            yl = dy[dx < 0]
            vidl = vid[dx < 0]

            yl_top = yl[yl>=0]
            yl_bot = yl[yl<0]
            vidl_top = vidl[yl>=0]#index
            vidl_bot = vidl[yl<0]

            # center
            xc = dx[dx >= 0]
            yc = dy[dx >= 0]
            vidc = vid[dx >= 0]
            yc = yc[xc < 200]
            vidc = vidc[xc < 200]
            xc = xc[xc < 200]

            yc_top = yc[yc>=0]
            yc_bot = yc[yc<0]
            vidc_top = vidc[yc>=0]
            vidc_bot = vidc[yc<0]

            # right
            xr = dx[dx >= 200]
            yr = dy[dx >= 200]
            vidr = vid[dx >= 200]

            yr_top = yr[yr>=0]
            yr_bot = yr[yr<0]
            vidr_top = vidr[yr>=0]
            vidr_bot = vidr[yr<0]


            # parameters
            mini_top = 7
            mini_bot = 6

            # left top
            iy = np.argsort(yl_top)
            iy = iy[0:min(mini_top, len(yl_top))]# 최대 6개의 좌측에 존재하는 장애물을 고려.
            yl_top = np.array([yl_top[i] for i in iy])
            vidl_top = np.array([vidl_top[i] for i in iy])
            # left bottom
            iy = np.argsort(yl_bot)
            iy = np.array(list(reversed(iy)))
            iy = iy[0:min(mini_bot, len(yl_bot))]
            yl_bot = np.array([yl_bot[i] for i in iy])
            vidl_bot = np.array([vidl_bot[i] for i in iy])

            # center top
            iy = np.argsort(yc_top)
            iy = iy[0:min(mini_top, len(yc_top))]
            yc_top = np.array([yc_top[i] for i in iy])
            vidc_top = np.array([vidc_top[i] for i in iy])
            # center bottom
            iy = np.argsort(yc_bot)
            iy = np.array(list(reversed(iy)))
            iy = iy[0:min(mini_bot, len(yc_bot))]
            yc_bot = np.array([yc_bot[i] for i in iy])
            vidc_bot = np.array([vidc_bot[i] for i in iy])

            # right top
            iy = np.argsort(yr_top)
            iy = iy[0:min(mini_top, len(yr_top))]
            yr_top = np.array([yr_top[i] for i in iy])
            vidr_top = np.array([vidr_top[i] for i in iy])
            # right bottom
            iy = np.argsort(yr_bot)
            iy = np.array(list(reversed(iy)))
            iy = iy[0:min(mini_bot, len(yr_bot))]
            yr_bot = np.array([yr_bot[i] for i in iy])
            vidr_bot = np.array([vidr_bot[i] for i in iy])


            #traj_v
            traj_class[k,8:14] = np.concatenate((np.zeros(6 - len(vidl_bot)),vidl_bot))#object_i의 left bottom에 위치한 다른 object들의 id 정보.
            traj_class[k,14:21] = np.concatenate((vidl_top ,np.zeros(7 - len(vidl_top))))#left top
            traj_class[k,21:27] = np.concatenate((np.zeros(6 - len(vidc_bot)),vidc_bot))#center bottom
            traj_class[k,27:34] = np.concatenate((vidc_top ,np.zeros(7 - len(vidc_top))))#center top
            traj_class[k,34:40] = np.concatenate((np.zeros(6 - len(vidr_bot)),vidr_bot))#right bottom
            traj_class[k,40:47] =np.concatenate((vidr_top ,np.zeros(7 - len(vidr_top))))#right top

    # convert from pixel to feet

    if homography_dir:#None
        traj = px_to_ft(traj, homography_dir)


    # create track
    ids = np.unique(traj[:,1])# Get all vehicle's id
    track = {} 
    for i in range(len(ids)):
        vtrack = traj[traj[:,1] == ids[i]]# get all row for 'ids[i]'
        track[ids[i]] = vtrack[:,2:5].T # get (t, x, y)


    # np.save(out_dir, np.array([traj, track]))
    np.save(out_dir, np.array([traj_class, track, traj]))

    # transfrom의 output
    '''
    traj  =  dsId.txt 파일과 같은 크기의 row. 각 row마다 [ dsId, object id(i), frame, x, y, i의 주변 39개까지의 주변 장애물과의 위치 관계]
    track =  dsId.txt 파일에 존재하는 obejct(i)의 trajectory {object_id(i): [ [frame_1, x_1, y_1], [frame_2, x_2, y_2], ... [frame_p, x_p, y_p]  ]}
    '''