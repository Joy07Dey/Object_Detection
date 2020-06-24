from __future__ import division

import cv2
import torch 
from torch.autograd import Variable
import time
import pickle as pkl
from utility import *
from darknet import Darknet

import tracemalloc
tracemalloc.start()



if __name__ == '__main__':
    print("............Object Detection has started..............")
    cfgfile = "yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
        
    num_classes = 80
    bbox_attrs = 5 + num_classes
	
    print("Loading Darknet............")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("....CNN loaded successfully")
	
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()            
    model.eval()    
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("................webcam started................")
	
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            
            img, orig_im, dim = prep_image(frame, inp_dim)
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()            
            
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
      

            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            classes = load_classes('coco.names')
            colors = pkl.load(open("colors", "rb"))
            
            list(map(lambda x: write(x, orig_im, classes, colors), output))
            
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
#            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
#            current, peak = tracemalloc.get_traced_memory()
#            print("Current memory usage is {" + str(current/ 10**6) +"}MB and Peak was {" + str(peak / 10**6) + "}MB")
            
        else:
            break
    
cap.release()
cv2.destroyAllWindows()
current, peak = tracemalloc.get_traced_memory()
#print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
print("Peak memory usage was " + str(peak / 10**6) + " MB")
print("................webcam stopped................")
print("............Object Detection has stopped..............")
tracemalloc.stop()    
    

