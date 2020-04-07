def pedestrians(path, w, h, n):
    from imutils.object_detection import non_max_suppression
    from imutils import paths
    import argparse
    import imutils
    from matplotlib import pyplot as plt
    import cv2
    import os
    import numpy as np
    import glob
    from tqdm import tqdm_notebook
    import pickle


    ### Create a list with the path of all the images in the file img1
    img_path = glob.glob(path + "/*.jpg")
    img_path.sort()
    img_path

    ### Background Subtraction of all the image in the folder and create image_bis
    print(" Background Subtraction of all the images in the folder ","\n")
    img_path_ = list()
    fgbg = cv2.createBackgroundSubtractorMOG2()
    if not os.path.exists('img1_bis'):
        os.mkdir('img1_bis')

    # Computing background
    for id_im, im_path in enumerate(img_path):
        print("Frame #" + str(id_im) + '/' + str(len(img_path)), end="\r")
        im = cv2.imread(im_path)
        fgmask = fgbg.apply(im)
        img_path_.append('img1_bis' + '/' + '{:03d}'.format(id_im)+ '.jpg')
        cv2.imwrite('img1_bis' + '/' + '{:03d}'.format(id_im)+ '.jpg', np.expand_dims((fgmask>0), axis=-1)*im)
    dic_Paths = dict((path_bis,path) for (path_bis,path) in zip(img_path_,img_path))


    ###pedestrian detection using HOG and SVM
    # initialize the HOG descriptor/person detector
    print("pedestrian detection using HOG and SVM","\n")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    dic_img_box = dict()

    for path in tqdm_notebook(img_path_):
        solution = []
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        orig = image.copy()

        #playing with winStride impact the performance, setting it to 7,7 allows us to have
            #a score of 24%
        (rects, weights) = hog.detectMultiScale(image, winStride=(7,7),padding=(8, 8), scale=1.05)
        #new_rects , new_weights = rects.copy(), weights.copy()
        Del =list()
        threshold = 0
        for i in range(len(rects)) :
            x,y,w,h = rects[i,:]
            if weights[i,0] < threshold:
                #cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 10)
                Del.append(i)
        rects , weights = np.delete(rects, Del, 0), np.delete(weights, Del,0)
            #cv2_imshow('Pedestrians', frame)
        #plt.imshow(orig)
        #plt.show()

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        dic_img_box[path] = pick

    ## Exporting as pickle
    pickle.dump(dic_img_box, open('dic_img_box_' + '{}'.format(threshold)+ '.p', "wb"))

    dic_img_box = pickle.load(open('dic_img_box_' + '{}'.format(threshold)+ '.p','rb'))


    ###Load the image, get the contour of the shape in it and check if they are human-shape like
    ###and returns boxes that could be human in shape
    print("Possible human regions using Contour detection ","\n")


    dic_img_human = dict()
    for path in tqdm_notebook(img_path_):
        im = cv2.imread(path)
        orig = im.copy()
        height, width = im.shape[:2]
        new_width = 500
        new_height = new_width*height//width
        im = cv2.resize(im,(new_width, new_height), interpolation = cv2.INTER_CUBIC)

        # Change to gray and apply both gaussian and threshold filter
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blurred_im = cv2.GaussianBlur(im_gray, (1, 1), 0)
        ret,thresh = cv2.threshold(blurred_im, 220, 255, 0)

        # Compute contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print( contours)
        # Get dimension of main contours
        human_boxes = []
        for cnt in contours:

            # Compute area size
            area = cv2.contourArea(cnt)
            if area > 3: #Chosen after studying are of tarpaulin
                # remove overdimension of contours

                cnt_low = cnt[:, 0]

                # contour width
                x_max = np.max(cnt_low[:, 0])*width//new_width
                x_min = np.min(cnt_low[:, 0])*width//new_width
                # contour height
                y_max = np.max(cnt_low[:, 1])*height//new_height
                y_min = np.min(cnt_low[:, 1])*height//new_height
                #cv2.rectangle(orig, (x_min, y_min), (x_max, y_max), (0, 255, 0), 10)
                human_boxes.append([x_min, y_min, x_max, y_max])

        #plt.imshow(orig)
        #plt.show()
        dic_img_human[path]= human_boxes

    ## Exporting as pickle
    pickle.dump(dic_img_human, open("dic_img_human.p", "wb"))

    dic_img_human = pickle.load(open('dic_img_human.p','rb'))

    print("Keeping the overlaping of the 2 sets of regions as final boxes   ","\n")

    def doOverlap(box1, box2):
    # Returns true if two rectangles(l1, r1)
    # and (l2, r2) overlap
        # If one rectangle is on left side of other
        if(box1[0] > box2[2] or box2[0] > box1[2]):
            return False

        # If one rectangle is above other
        if(box1[1] >box2[3] or box2[1] > box1[3]):
            return False

        return True

    if not os.path.exists('img1_boxes'):
        os.mkdir('img1_boxes')
    dic_final_boxes = dict()
    for (id_im,path) in tqdm_notebook(enumerate(dic_img_box.keys())):
        pick = dic_img_box[path]
        pick_ = np.copy(pick)
        Del= list()
        for i,box1 in enumerate(pick):
            overlap = [doOverlap(box1, box2) for box2 in dic_img_human[path]]
            if sum(overlap)==0:
                Del.append(i)
        pick_=np.delete(pick_,Del,0)
        dic_final_boxes[dic_Paths[path]] = pick_
        image = cv2.imread(dic_Paths[path], cv2.IMREAD_UNCHANGED)
        for (xA, yA, xB, yB) in pick_:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 10)
        cv2.imwrite('img1_boxes' + '/' + '{:03d}'.format(id_im)+ '.jpg', image)

    ## Exporting as pickle

    pickle.dump(dic_final_boxes, open('dic_final_boxes_' + '{}'.format(threshold)+ '.p', "wb"))
    dic_final_boxes = pickle.load(open('dic_final_boxes_' + '{}'.format(threshold)+ '.p','rb'))

    bounding_boxes = list()
    for frame_id, frame_path in enumerate(list(img_path)):
        for bb_id, box in enumerate(dic_final_boxes[frame_path]):
            bounding_boxes.append([frame_id, bb_id, box[0], box[1], box[2]-box[0],box[3]-box[1]])
    return(bounding_boxes)
