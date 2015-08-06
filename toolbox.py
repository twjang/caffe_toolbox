#!/usr/bin/env python

CAFFE_ROOT = '/home/nyamnyam/Development/caffe/'

import sys
import os
sys.path.append(os.path.join(CAFFE_ROOT, 'python'))

import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import base64
import time
import scipy as sp
import scipy.io
import cv2
import h5py
import statsmodels.api as sm

caffe.set_mode_gpu()

COMMON_CODE = """
import sys
import os
sys.path.append('""" +os.path.join(CAFFE_ROOT, 'python')+ """')

import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import base64
import cv2
import h5py
import statsmodels.api as sm
import random
import time

caffe.set_mode_gpu()

## Taken from http://nbviewer.ipython.org/github/bvlc/caffe/blob/master/examples/filter_visualization.ipynb

def cdf(data):
    tmp = np.ma.masked_array(data,np.isnan(data))
    tmp = tmp.reshape((tmp.size,))
    x = np.linspace(tmp.min(), tmp.max())
    ecdf = sm.distributions.ECDF(tmp)
    y = ecdf(x)
    diff_y = - (np.hstack([[0],y]) - np.hstack([y,[0]]))[1:y.size] / (x[1]-x[0])
    diff_x = x[1:]
    plt.figure()
    plt.subplot(1,2,1)
    plt.step(x, y)
    plt.subplot(1,2,2)
    plt.step(diff_x, diff_y)
    plt.show()


def mkarr(data):
    data = np.array(data.data).copy().reshape(data.shape)
    return data

def show(data, padsize=1, padval=None, defcm=cm.Greys_r):
    first = True

    if isinstance(data, list):
        datalst = data
    else:
        if len(data.shape)==4:
            datalst = [data[i,:,:,:] for i in xrange(data.shape[0])]
        else: datalst = [data]

    if len(datalst) > 10:
        print "You're now plotting %d images. Continue?(y/N)" % len(datalst), 
        sure = raw_input()
        if sure.strip().lower() != 'y':
            return

    for data in datalst:
        tmp = np.ma.masked_array(data,np.isnan(data))
        curmin = tmp.min()
        curmax = tmp.max()
        curzero = (curmin  + curmax) / 2.0
        if padval is None: pval = curzero
        else: pval = padval

        print "min = ", curmin
        print "max = ", curmax
        print "contains NaN = ", data.max() != curmax
        
        del tmp
            
        
        if len(data.shape)==3:
            # force the number of filters to be square
            n = int(np.ceil(np.sqrt(data.shape[0])))
            padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
            data = np.pad(data, padding, mode='constant', constant_values=(pval, pval))
            
            # tile the filters into an image
            data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
            data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])


        if not first: plt.figure()
        plt.imshow(data,interpolation="nearest", cmap=defcm, vmin = data.min(), vmax = data.max())

        first = False

    plt.show()

def lslayer(curnet=None):
    global net
    if curnet is None: curnet = net
    for idx in xrange(len(curnet.layers)):
        layer_name = curnet._layer_names[idx]
        layer_type = curnet.layers[idx].type
        layer_blob_num = len(curnet.layers[idx].blobs)
        print "layer[%d]:%s, %s" % (idx, layer_name, layer_type)
        for i in xrange(layer_blob_num):
            print "    blob[%d]: %s" % (i, str(curnet.layers[idx].blobs[i].data.shape))

def lsblob(curnet=None):
    global net
    if curnet is None: curnet = net
    for key in net.blobs.keys():
        print key, ":", net.blobs[key].data.shape

def bb(*kargs):
    global net
    
    kargs = list(kargs)
    diff = False

    if 'diff' in kargs:
        kargs.remove('diff')
        diff = True
    if len(kargs) < 1: raise "See usage by typing hlp()"
    if isinstance(kargs[0], caffe._caffe.Net):
        curnet = kargs[0]
        try: idx1 = kargs[1]
        except IndexError: idx1 = None
        try: idx2 = kargs[2]
        except IndexError: idx2 = None
    else:
        curnet = net
        try: idx1 = kargs[0]
        except IndexError: idx1 = None
        try: idx2 = kargs[1]
        except IndexError: idx2 = None

    if not idx2 is None:
        if diff: return net.layers[idx1].blobs[idx2].diff
        else: return net.layers[idx1].blobs[idx2].data
    else:
        if diff: return net.blobs[idx1].diff
        else: return net.blobs[idx1].data

def dd(*kargs):
    kargs=list(kargs)
    kargs.append('diff')
    return bb(*kargs)

def go(*kargs): 
    if len(kargs) == 0:
        use_net=None
        data=None
    elif len(kargs) == 1:
        if not isinstance(kargs, caffe.Net):
            data = kargs[0]
        else: use_net = kargs[0]
    elif len(kargs) == 2:
        use_net = kargs[0]
        data = kargs[1]
    else:
        print "See usage by typing hlp()"
        return
    global net
    if use_net is None: use_net = net
    if not data is None:
        pass
    use_net.forward()
    use_net.backward()

def fg(use_net=None):
    global net
    if use_net is None: use_net = net
    use_net.forward()

def bg(use_net=None):
    global net
    if use_net is None: use_net = net
    use_net.backward()

def bloblst(use_net=None):
    global net
    if use_net is None: use_net = net
    return net.blobs.keys()

def layerlst(curnet=None):
    res = []
    global net
    if curnet is None: curnet = net
    for idx in xrange(len(curnet.layers)):
        layer_name = curnet._layer_names[idx]
        layer_type = curnet.layers[idx].type
        layer_blob_num = len(curnet.layers[idx].blobs)
        if layer_blob_num > 0:
            res.append(idx)
    return res

def showrange(data):
    first = True
    if isinstance(data, list):
        datalst = data
    else:
        datalst = [data]

    for data in datalst:
        tmp = np.ma.masked_array(data,np.isnan(data))
        curmin = tmp.min()
        curmax = tmp.max()

        print "min = %15e  max=%15e  NaN=%s" % (curmin, curmax, str(data.max() != curmax))

def dbgsgd2(curnet=None):
    global net
    if curnet is None: curnet = net

    for i in curnet.blobs.keys():
        print " == blob[%s]" % (i)
        
        data = curnet.blobs[i].data
        tmp = np.ma.masked_array(data,np.isnan(data))
        curmin = tmp.min()
        curmax = tmp.max()
        curvar = tmp.var()
        print "      sz: min=%+10e  max=%+10e  var=%+10e  NaN=%s" % (curmin, curmax, curvar, str(data.max() != curmax) )

        data = curnet.blobs[i].diff
        tmp = np.ma.masked_array(data,np.isnan(data))
        curmin = tmp.min()
        curmax = tmp.max()
        curvar = tmp.var()
        print "    grad: min=%+10e  max=%+10e  var=%+10e  NaN=%s" % (curmin, curmax, curvar, str(data.max() != curmax) )
        print ""


def dbgsgd(curnet=None):
    global net
    if curnet is None: curnet = net
    for idx in xrange(len(curnet.layers)):
        layer_name = curnet._layer_names[idx]
        layer_type = curnet.layers[idx].type
        layer_blob_num = len(curnet.layers[idx].blobs)
        if layer_blob_num == 0: continue
        print ""
        print "layer[%d]:%s (%s)" % (idx, layer_name, layer_type)

        for i in xrange(layer_blob_num):
            print " == blob[%d]" % (i)
            
            data = curnet.layers[idx].blobs[i].data
            tmp = np.ma.masked_array(data,np.isnan(data))
            curmin = tmp.min()
            curmax = tmp.max()
            print "      sz: min=%+10e  max=%+10e  NaN=%s" % (curmin, curmax, str(data.max() != curmax) )

            data = curnet.layers[idx].blobs[i].diff
            tmp = np.ma.masked_array(data,np.isnan(data))
            curmin = tmp.min()
            curmax = tmp.max()
            print "    grad: min=%+10e  max=%+10e  NaN=%s" % (curmin, curmax, str(data.max() != curmax) )
    

def hlp():
    print "* hlp(): show help"
    print ""
    print "* go(net): run 1 iteration"
    print "* fg(net): forward"
    print "* bg(net): backward" 
    print ""
    print "* mkarr(blob): make blob into np.array"
    print "* show(data, padsize=1, padval=0): draw 4D-array whose shape is (numch, patch_h, patch_w)"
    print ""
    print "* lslayer(net[default by net]): list all layers"
    print "* lsblob(net[default by net]): list all blobs" 
    print ""
    print "* bb(net[default by net], blob_name): returns blob"
    print "* bb(net[default by net], layeridx, blob_name): returns blob associated to layer[layeridx]"
    print "* dd(net[default by net], blob_name): returns diff"
    print "* dd(net[default by net], layeridx, blob_name): returns diff associated to layer[layeridx]"
    print ""
    print "* bloblst(net[default by net]): returns the list of all blob keys"
    print "* layerlst(net[default by net]): returns the list of indices of all layers with weights"
    print ""
    print "* showrange(data): prints range of given matrices"
    print "* dbgsgd(net[default by net]): debug learning rate"
    print "* dbgsgd2(net[default by net]): debug learning rate"
    print "" 
    print "* cdf(data): draw sample cdf function"
"""

def randstr(length):
    return ''.join([chr(random.randint(0, 10) + ord('a')) for i in xrange(length)])

def do_extract(args):
    if len(args)<3:
        print ("Usage: [prototxt] [caffemodel] [output file name]")
        exit(-1)

    fname_prototxt = args[0]
    fname_caffemodel = args[1]
    fname_output = args[2]

    net = caffe.Net(fname_prototxt, fname_caffemodel, caffe.TEST)    
    layers = net.params.keys()
    final = {}
    for layer in layers:
        final[layer+'_weight'] = net.params[layer][0].data
        final[layer+'_bias']   = net.params[layer][1].data
    if fname_output.endswith('.mat'):
        scipy.io.savemat(fname_output, final)
    elif fname_output.endswith('.hdf') or fname_output.endswith('.h5') or fname_output.endswith('.hdf5'):
        f = h5py.File(fname_output, 'w')
        for key in final.keys():
            f[key]=final[key]
        f.close()


def do_clean(args):
    os.system("rm -rf ./snapshot/*")
    os.system("rm -f ./train.log")

def do_dataset(args):
    os.system("rm -rf ./dataset/*")
    os.system("./create_chinese.sh")
    do_mean()

def do_pltacc(args):
    plt.ion()
    fname = args[0]

    if len(args)>1: recent_entry = int(args[1])
    else: recent_entry = None

    if len(args)>2: smooth = int(args[2])
    else: smooth = None

    while True:
        f=file(fname, 'r')
        lines=f.readlines()
        f.close()

        iter_indices = []
        val = []
        idx = 0 
        for line in lines:
            if line.find('Test net') > -1 and line.find('accuracy') > -1:
                v = line.strip().split('accuracy = ')[1]
                iter_indices.append(float(idx))
                val.append(float(v))
                idx += 1
        y = np.array(val)
        x = np.array(iter_indices)
        if not recent_entry is None:
            y = y[-recent_entry:]
            x = x[-recent_entry:]
        plt.clf()
        plt.plot(x, y)
        if not smooth is None:
            yy = moving_average(y, smooth)
            xx = moving_average(x, smooth)
            plt.plot(xx, yy)
        plt.pause(2.0)

def moving_average(a, n=3) :
    ret = np.cumsum(np.array(a), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def do_pltloss(args):
    plt.ion()
    fname = args[0]


    if len(args)>1: recent_entry = int(args[1])
    else: recent_entry = None
    
    if recent_entry == -1: resent_entry = None

    if len(args)>2: smooth = int(args[2])
    else: smooth = None
    if smooth == -1: smooth = None
    
    if len(args)>3: title = args[3]
    else: title = "test acc. / trng loss"


    while True:
        f=file(fname, 'r')
        lines=f.readlines()
        f.close()

        ## draw accuracy
        iter_indices = []
        val = []
        idx = 0 
        for line in lines:
            if line.find('Test net') > -1 and line.find('accuracy') > -1:
                v = line.strip().split('accuracy = ')[1]
                iter_indices.append(float(idx))
                val.append(float(v))
                idx += 1
        y = np.array(val)
        x = np.array(iter_indices)
        if not recent_entry is None:
            y = y[-recent_entry:]
            x = x[-recent_entry:]
       
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
        plt.clf()
        plt.subplot(1,2,1)
        line1 = plt.plot(x, y)

        if not smooth is None:
            yy = moving_average(y, smooth)
            xx = moving_average(x, smooth)
            line2 = plt.plot(xx, yy)
        
        
        if not smooth is None:
            plt.setp(line1, 'color', '#aaaaaa')
            plt.setp(line2, 'color', '#ff0000')


        ## draw loss        
        iter_indices = []
        val = []

        for line in lines:
            if line.find('Iteration') > -1 and line.find('loss') > -1:
                v = line.strip().split('loss = ')[1]
                i = line.strip().split('Iteration ')[1].split(',')[0]
                iter_indices.append(float(i))
                val.append(float(v))
        y = np.array(val)
        x = np.array(iter_indices)
        if not recent_entry is None:
            y = y[-recent_entry:]
            x = x[-recent_entry:]

        plt.subplot(1,2,2)
        line1 = plt.plot(x, y)

        if not smooth is None:
            yy = moving_average(y, smooth)
            xx = moving_average(x, smooth)
            line2=plt.plot(xx, yy)

        if not smooth is None:
            plt.setp(line1, 'color', '#aaaaaa')
            plt.setp(line2, 'color', '#ff0000')
        plt.pause(1.0)
        
def do_eval(args):
    if len(args)<2:
        print "arg1 = model.prototxt, arg2 = model.caffemodel, arg3 = report(optional)"
        return

    model_def = args[0]
    model_fname = args[1]
    if len(args) > 2: report_path = args[2]
    else: report_path = None

    net = caffe.Net(model_def, model_fname, caffe.TEST) 
    print "loaded"

    class_lbl="0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"

    def remap(idx):
        if idx < 10: return idx
        return 10 + (idx - 10) / 2
    def cls2char(cls):
        return class_lbl[cls]

    confusion_matrix = np.zeros([62,62])
    confusion_matrix_insensitive = np.zeros([36,36])


    answers = []
    answers_by_cls = {}

    for i in xrange(100):
        net.forward()
        label = list(net.blobs['label'].data.astype(np.int64))
        result = list(net.blobs['fc7'].data.argmax(1))
        for lbl, out in zip(label, result):
            confusion_matrix[lbl, out] += 1.0
            confusion_matrix_insensitive[remap(lbl), remap(out)] += 1.0

        for idx in xrange(len(result)):
            ent = (net.blobs['data'].data[idx].copy(), label[idx], result[idx])
            answers.append(ent)

    for ent in answers:
        img, lbl, res = ent
        if (answers_by_cls.has_key(lbl)):
            isok = 1 if lbl == res else 0
            answers_by_cls[lbl].append((isok, (img, res)))
        else:
            isok = 1 if lbl == res else 0
            answers_by_cls[lbl] = [(isok, (img, res)),]
    
    for key in answers_by_cls.keys():
        entries = answers_by_cls[key]
        entries.sort(key=lambda e: e[0])
        answers_by_cls[key] = entries
                
    if report_path:
        try: os.makedirs(report_path)
        except OSError: pass
        f = file(os.path.join(report_path, 'index.html'), 'w')
        f.write("""
            <!DOCTYPE html>
            <html>

            <frameset cols="25%,*">
            <frame src="menu.htm">
            <frame name="view">
            </frameset>

            </html>""")
        f.close()

        fmenu = file(os.path.join(report_path, 'menu.htm'), 'w')
        fmenu.write('<html><head></head><body>')

        for cls in answers_by_cls.keys():
            fmenu.write('<a href="./cls_%d.html" target="view">%d</a><br/>\n' % (cls, cls))
            try: os.makedirs(os.path.join(report_path, 'imgs_%d' % cls))
            except OSError: pass
            idxcnt = 0
            idxfname = os.path.join(report_path, 'cls_%d.html' % cls)
            idxf = file(idxfname, 'w')
            idxf.write('<html><head>')
            idxf.write('<style>.item { border:1px solid #000; display:inline-block; width:150px;}\n')
            idxf.write('.green { color:#20EE20; font-weight:bold;}\n')
            idxf.write('.wrong { color:#EE2020; font-weight:bold;}\n')
            idxf.write('</style>')
            idxf.write('</head><body>')
            idxf.write('<h1>%d</h1>\n' % cls)
            for isok, data in answers_by_cls[cls]:
                img, res = data
                idxcnt += 1
                img -= img.min()
                img /= img.max()
                img = img[0]
                cv2.imwrite(os.path.join(report_path, 'imgs_%d' % cls, '%d.png' % idxcnt), img * 255)

                if res == cls: anscolor = 'green'
                else: anscolor = 'wrong'
                idxf.write('<div class="item"><img src="./imgs_%d/%d.png" /><span class="%s">&#%d;</span>(&#%d;)</div>\n' % (cls, idxcnt, anscolor, ord(cls2char(res)), ord(cls2char(cls))))
            idxf.write('</body></html>')
            idxf.close()

        fmenu.write('</body></html>') 
        fmenu.close()

    res1 = confusion_matrix / confusion_matrix.sum(1).reshape(62, 1)
    res2 = confusion_matrix_insensitive / confusion_matrix_insensitive.sum(1).reshape(36, 1)
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(res1, interpolation='nearest')
    plt.subplot(2,1,2)
    plt.imshow(res2, interpolation='nearest')
    print "sensitive:   ", np.trace(res1) / res1.shape[0]
    print "insensitive: ", np.trace(res2) / res2.shape[0]

    plt.show()


def do_ipython(args):
    global CAFFE_ROOT
    code = """
import sys
import os
sys.path.append('""" + os.path.join(CAFFE_ROOT, 'python') + """')

import caffe
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
"""
    tmpfname = '/tmp/' +randstr(10) 
    f=file(tmpfname, 'w') 
    f.write(code)
    f.close()

    os.system('ipython -i %s' % tmpfname)
    os.system('rm -f %s' % tmpfname)

def do_mean(args):
    global CAFFE_ROOT
    os.system("""
        export CAFFE_PATH=\"""" + CAFFE_ROOT + """\"
        $CAFFE_PATH./build/tools/compute_image_mean -backend lmdb ./dataset/chinese_train_lmdb ./dataset/chinese_mean.binaryproto
        echo "Done."
        """)


def do_train(args):
    global CAFFE_ROOT
    os.system("""
        export CAFFE_PATH=\"""" + CAFFE_ROOT + """\"
        $CAFFE_PATH/./build/tools/caffe train  --solver=solver.prototxt 2>&1 | tee -a train.log
        #./learn train ./charconf.ini  --solver=solver.prototxt 2>&1 | tee -a train.log
    """ )



def do_resume(args):
    global CAFFE_ROOT
    if len(args)<1:
        print "Please specify the .solverstate file"
        return

    fname = args[0]
    os.system("""
        export CAFFE_PATH=\"""" + CAFFE_ROOT + """\"
        ./learn train  --solver=solver.prototxt \
        --snapshot=%s
    """% (fname))


def do_loadstate(args):
    global COMMON_CODE
    if len(args)<1:
        print "Please specify the .caffemodel file"
        return 

    code = COMMON_CODE + """
model_def = base64.b64decode('%s')
model_fname = base64.b64decode('%s')
net = caffe.Net(model_def, model_fname, caffe.TEST) 
del model_fname
del model_def

print " *** net_param is set for you"
print " *** to plot the weight, "
print 
print "d = mkarr(bb(2,0))"
print "show(d)"

hlp()
""" % (base64.b64encode(args[0]), base64.b64encode(args[1]))
    tmpfname = '/tmp/' +randstr(10) 
    f=file(tmpfname, 'w') 
    f.write(code)
    f.close()

    os.system('ipython -i %s' % tmpfname)
    os.system('rm -f %s' % tmpfname)




def do_pltblob(args):
    if len(args)>0: fname = args[1]
    else: fname = './dataset/chinese_mean.binaryproto'

    data = file(fname,'r').read()
    b=caffe.io.caffe_pb2.BlobProto()
    b.ParseFromString(data)
    arr = caffe.io.blobproto_to_array(b)

    c=arr.shape[1]
    h=arr.shape[2]
    w=arr.shape[3]

    res = np.hstack([arr[0, 0, :,:].reshape((h,w)), arr[0, 1, :,:].reshape((h,w)), arr[0, 2, :,:].reshape((h,w))])
    myarr = np.zeros((h, w, c))
    for idx in xrange(c): myarr[:,:,idx] = arr[0, idx, :, :].reshape((h, w))

    plt.subplot(211)
    plt.imshow(res)
    plt.subplot(212)
    plt.imshow(myarr)
    plt.show()

def do_draw(args):
    global CAFFE_ROOT
    if len(args)>0: fname = args[0]
    else: fname='./model_train.prototxt'

    os.system(os.path.join(CAFFE_ROOT, '/python/draw_net.py') + ' %s /tmp/net.png' % fname)
    os.system('shotwell /tmp/net.png')

def do_pack(args):
    if (len(args) < 1):
        print "usage: [pack dir name]"
        return
    
    d = args[0]
    try: os.makedirs(d)
    except OSError: pass
    
    os.system('cp ./snapshot/ "./%s/" -R' % d)
    os.system('cp ./model_train.prototxt "./%s/" ' % d)
    os.system('cp ./train.log "./%s/" ' % d)
    os.system('cp ./solver.prototxt "./%s/" ' % d)


def do_visevol(args):
    if (len(args) < 3):
        print "usage: [model def] [snapshot dir] [output dir]"
        return
    
    model_def = args[0]
    model_dir = args[1]
    path = args[2]
    
    snapshots = []
    for c in os.listdir(model_dir):
        if c.endswith('.caffemodel'):
            d = c[:-len('.caffemodel')]
            iternum = int(d.split('_')[-1])
            snapshots.append((iternum, os.path.join(model_dir, c)))

    snapshots.sort()
    for iternum, model_fname in snapshots:
        steppath = os.path.join(path, 'iter_%d' % iternum)
        if not os.path.exists(steppath):
            do_visweight([model_def, model_fname, steppath, 'index.html' ])
    
    f = file(os.path.join(path, 'index.html'), 'w')
    f.write("""
        <!DOCTYPE html>
        <html>

        <frameset cols="25%,*">
        <frame src="menu.htm">
        <frame name="view">
        </frameset>

        </html>""")
    f.close()
    f = file(os.path.join(path, 'menu.htm'), 'w')
    f.write('<html><head></head><body>')
    for iternum, model_fname in snapshots:
        f.write('<a href="./iter_%d/index.html" target="view">%d</a><br/>\n' % (iternum, iternum))
    f.write('</body></html>') 
    f.close()

def do_evolmovie(args):
    if (len(args) < 3):
        print "usage: [visevol output dir] [layer id] [output file]"
        return

    visevol_path = args[0]
    layerid = args[1]
    outfname = args[2]
    
    imglst = []
    for c in os.listdir(visevol_path):
        curdir = os.path.join(visevol_path, c)
        if c.startswith('iter_') and os.path.isdir(curdir):
            imgdir = os.path.join(curdir, 'imgs')
            imgpath = os.path.join(imgdir, layerid + '.jpg')
            iternum = int(c.replace('iter_', ''))
            if os.path.isfile(imgpath):
                imglst.append((iternum, imgpath))
            else:
                print "No such layer id(%s)" % layerid
                print "Please use .. "
                for f in os.listdir(imgdir):
                    if not f.endswith('.jpg'): continue
                    print f.replace('.jpg', '')
                return
    
    if len(imglst) == 0:
        print "Please check your arguments: no images found"
        return

    imglst.sort()

    im = cv2.imread(imglst[0][1])
    isColor = 1
    fps = 15
    frameW = im.shape[1]
    frameH = im.shape[0]

    print frameH, "x", frameW

    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(outfname, fourcc, fps, (frameW, frameH)) 
    for iternum, imgf in imglst:
        im = cv2.imread(imgf)
        cv2.putText(im, '%10d' % iternum, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0)) 
        writer.write(im)

    cv2.destroyAllWindows()
    writer.release()
    writer = None
    print "Done"


def do_visweight(args):
    if (len(args) < 3):
        print "usage: [model def] [caffemodel] [output dir] [output fname]"
        return
    
    model_def = args[0]
    model_fname = args[1]
    path = args[2]
    imgpath = os.path.join(path, 'imgs')
    if len(args)>3: output_fname = os.path.join(path, args[3])
    else: output_fname = os.path.join(path, 'index.html')
    
    try: os.makedirs(path)
    except OSError: pass
    try: os.makedirs(imgpath)
    except OSError: pass

    ## Weight development

    net = caffe.Net(model_def, model_fname, caffe.TEST) 
    
    f = file(output_fname, 'w')
    f.write('<html>') 
    f.write('<head>') 
    f.write('</head>') 
    f.write('<body>') 



    for idx in xrange(len(net.layers)):
        layer_blob_num = len(net.layers[idx].blobs)
        layer_name = net._layer_names[idx]
        if layer_blob_num > 0:

            f.write('<h3>%s</h3>'% layer_name)
            weightdata = net.layers[idx].blobs[0].data

            if len(weightdata.shape)==4:
                imgfname = 'w%02d.jpg' % (idx)
                curweight_imgpath = os.path.join(imgpath, imgfname)
                curweight_relimgpath = os.path.join('imgs', imgfname)
                f.write('<img src="%s" />' % curweight_relimgpath)
                

                ### create weight image for each input channels
                gg = []
                
                for widx in xrange(weightdata.shape[1]):
                    data = weightdata [:,widx]

                    padsize = 1
                    padval = data.mean()

                    # force the number of filters to be square
                    n = int(np.ceil(np.sqrt(data.shape[0])))
                    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
                    data = np.pad(data, padding, mode='constant', constant_values=(0, padval))
                    
                    # tile the filters into an image
                    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
                    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

                    data -= data.min()
                    data /= data.max()

                    gg.append(data)


                gg = np.array(gg) 

                ### we create the final output image by remapping the collected weight images
                data = gg
                    
                padsize = 4
                padval = data.mean()

                # force the number of filters to be square
                n = int(np.ceil(np.sqrt(data.shape[0])))
                padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
                data = np.pad(data, padding, mode='constant', constant_values=(0, padval))
                
                # tile the filters into an image
                data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
                data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

                data -= data.min()
                data /= data.max()

                cv2.imwrite(curweight_imgpath, data * 255)


            elif len(weightdata.shape)==2:
                imgfname = 'w%02d_fc.jpg' % (idx)
                curweight_imgpath = os.path.join(imgpath, imgfname)
                curweight_relimgpath = os.path.join('imgs', imgfname)
                f.write('<img src="%s" />' % curweight_relimgpath)
                data = weightdata
                data -= data.min()
                data /= data.max()
                cv2.imwrite(curweight_imgpath, data * 255)
            f.write('<hr/>')

    f.write('</body>') 
    f.write('</html>') 
    f.close()

def do_spec(args):
    global CAFFE_ROOT
    fname = os.path.join(CAFFE_ROOT, 'src/caffe/proto/caffe.proto')
    os.system('less %s' % fname)
    return

COMMANDS=[
    ('clean',do_clean),
    ('dataset' ,do_dataset),
    ('ipython' ,do_ipython),
    ('mean'    ,do_mean),
    ('train'   ,do_train),
    ('resume'  ,do_resume),
    ('pltblob' ,do_pltblob),
    ('pltloss' ,do_pltloss),
    ('pltacc' ,do_pltacc),
    ('loadstate' ,do_loadstate),
    ('draw', do_draw),
    ('eval', do_eval),
    ('extract', do_extract),
    ('pack', do_pack),
    ('visweight', do_visweight),
    ('visevol', do_visevol),
    ('evolmovie', do_evolmovie),
    ('spec', do_spec),
]


def main():
    if len(sys.argv)<2:
        print "Usage: %s [command]" % sys.argv[0]
        for cmd, fun in COMMANDS:
            print "   ", cmd
        exit(0)

    for cmd, fun in COMMANDS:
        if sys.argv[1].strip().lower() == cmd:
            fun(sys.argv[2:])
            exit(0)

    print "Invalid command, availible commands are:"
    for cmd, fun in COMMANDS:
        print "   ", cmd

if __name__=="__main__":
    main()

