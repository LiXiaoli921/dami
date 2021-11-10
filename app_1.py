# encoding=utf-8
"""
Copyright (c) 2018 Heils.cn
Written by kohill.
"""
from __future__ import print_function
from __future__ import division
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import logging
import time
import os
import glob
import cv2
import threading
import tqdm
# Do not import any DL framework here, including tensorflow, torch, etc.
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue
import numpy as np
from easydict import EasyDict


class Config():
    logger = logging.getLogger('statisticNew')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('test.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    def getLog(self):
        return self.logger

conf = Config()
logger = conf.getLog()
logger.info('start')





def start_web_server(g_cfg, qe_in, da_dict, da_manager):
    # type: (EasyDict, Queue, dict, SyncManager, int) -> Any
    """
    :param g_cfg: A EasyDict to store global configuration, eg. gpu_id.
    :param qe_in: A queue to communicate id of each image, sending a id into this queue means data of the image is OK.
    :param da_dict: A shared dict created by SyncManager to share image data among sub-processes with shared memory.
    :param da_manager: Once an image is received, a shared dict is created by da_manager.
    :param port: The port you want to use.
    :return:
    """
    from gevent import pywsgi
    import gevent
    import flask
    import json
    from PIL import Image
    from io import StringIO as StringIO # new version of modeul of "cStringIO"
    import copy
    import gc
    # gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_LEAK)
    logging.info("web_server is initializing...")

    app = flask.Flask(__name__)
    gpu_id_output = g_cfg.gpu_id_output

    def index():
        return json.dumps({"Tensorflow": "unimplemented.", "G": gpu_id_output})

    app.add_url_rule(rule='/', view_func=index, methods=["GET", "POST"])
    lock = threading.Lock()
    import numpy as np
    def registered_user():
        if len(da_dict) > 20:
            return json.dumps({"S": 1, "G": gpu_id_output, "T": 0, "R": {}})
        global im_count_as_id
        with lock:
            im_count_as_id = im_count_as_id + 1
            local_im_count_as_id = copy.deepcopy(im_count_as_id)
        t_start = time.time()
        get_file = flask.request.files['file']
        pic = get_file.read()

        user_image = Image.open(StringIO(pic))
        user_image = np.asarray(user_image)
        im_da = da_manager.dict()
        im_da["data"] = user_image
        im_da["over"] = False
        da_dict[local_im_count_as_id] = im_da
        qe_in.put(local_im_count_as_id)
        # print (local_im_count_as_id)
        while True:
            gevent.sleep(0.01)
            im_da = da_dict[local_im_count_as_id]
            if im_da["over"] is True:
                # This means that this image has been processed.
                r = da_dict.pop(local_im_count_as_id)  # Delete the results from memory.
                from pprint import pprint
                # character_results = r["character_results"]
                context = r["context"]
                html_content = r['html_content']
                # for a in character_results:
                #     print(a)
                r.clear()
                del r
                success = 0
                break
        return json.dumps({"S": success, "G": gpu_id_output, "T": time.time() - t_start, "R": context,"H":html_content})

    app.add_url_rule(rule='/recog', view_func=registered_user, methods=["POST"])
    app.debug = False
    wsgi = pywsgi.WSGIServer(('0.0.0.0', global_config.port), app, error_log=None)
    wsgi.serve_forever()
    # app.run("0.0.0.0", debug=False, threaded=False, port=global_config.port)
    # http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    # http_server.listen(port, address="0.0.0.0")
    # logging.info("Tornado server starting on port {}".format(port))
    # tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    # test()
    # set some necessary environment values here
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        '-n', '--number',
        help="option a number to run a thing",
        type='int', default=0)
    opts, args = parser.parse_args()

    # global gpu_id
    gpu_id = opts.number
    port = 8120 + opts.number


    # Avoid deadlock
    cv2.setNumThreads(1)

    # Initialize the logging module.
    logging.basicConfig(level=logging.WARNING)

    # Global config
    global_config = EasyDict()
    global_config.gpu_id_output = opts.number

    global_config.gpu_id = 0
    global_config.fifo_maxsize = 3
    global_config.fifo_in_maxsize = 15
    # global_config.path_yml = 'Detection/text.yml'
    global_config.path_cfg_anchor_list = ['Detection/592_1200.npy', 'Detection/1200_592.npy']
    global_config.path_ckpt = 'Detection/ctpn_ep17_cls0.060644505565_regr0.0684062405247_loss0.129050746054_lstm.pth.tar'
    global_config.resume = 'flip_classify/trained_model/weights-16-160-[1.0000].pth'
    global_config.new_width = 1024
    global_config.port = port

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    processes = []
    # A dictionary which is used to share data among all processes.
    manager = SyncManager()
    manager.start()
    data_dict = manager.dict()  # type: dict

    # Start process for Text detection.
    q_text_detector = Queue(maxsize=global_config.fifo_maxsize)
    queue_construct_text_lines = Queue(maxsize=global_config.fifo_maxsize)
    p = Process(target=worker_text_detector,
                args=(data_dict, global_config, q_text_detector, queue_construct_text_lines))
    p.daemon = True
    p.start()
    processes.append(p)

    # Start process for construct_text_lines detection.
    q_text_recognizer = Queue(maxsize=global_config.fifo_maxsize)
    p = Process(target=worker_construct_text_lines, args=(data_dict, queue_construct_text_lines, q_text_recognizer))
    p.daemon = True
    p.start()
    processes.append(p)

    # Start process for Image pre-processing.
    q_in = Queue(maxsize=global_config.fifo_in_maxsize)
    q_in_formula_detection = Queue(maxsize=global_config.fifo_maxsize)
    p = threading.Thread(target=worker_image_pre_process, args=(data_dict, global_config, q_in, q_in_formula_detection))
    p.daemon = True
    p.start()
    processes.append(p)

    # Start process for Formula detection.
    p = threading.Thread(target=worker_formula_detector,
                         args=(data_dict, global_config, q_in_formula_detection, q_text_detector))
    p.daemon = True
    p.start()
    processes.append(p)

    # Start process for Text recognition.
    q_out = Queue(maxsize=global_config.fifo_maxsize)
    p = threading.Thread(target=worker_text_recognizer, args=(data_dict, global_config, q_text_recognizer, q_out))
    p.daemon = True
    p.start()
    processes.append(p)


    def _worker_feed_thread():
        im_id = 0
        while True:
            for im_name in glob.iglob("/home/ai/Downloads/1116/test_pic/*.*"):
                im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                im_da = manager.dict()
                im_da["data"] = im
                im_da["over"] = False
                data_dict[im_id] = im_da
                q_in.put(im_id)
                im_id += 1

        # This will terminal all processes.
        q_in.put(None)


    def _worker_read_thread():
        bar = tqdm.tqdm()
        while True:
            d = q_out.get()
            if d is None:
                for p in processes:
                    p.terminate()
                exit()
            im_da = data_dict[d]
            im_da["over"] = True
            data_dict[d] = im_da
            # bar.update(1)

    def _worker_read_thread_debug():
        bar = tqdm.tqdm()
        while True:

            d = q_out.get()
            if d is None:
                for p in processes:
                    p.terminate()
                exit()
            context = data_dict.pop(d)["context"] # type: dict
            for x in context:
                print(x)
            bar.update(1)


    t_read = threading.Thread(target=_worker_read_thread, args=())
    t_read.daemon = True
    t_read.start()
    #
    # t_feed = threading.Thread(target=_worker_feed_thread, args=())
    # t_feed.daemon = True
    # t_feed.start()
    start_web_server(global_config, q_in, da_dict=data_dict, da_manager=manager)
    t_read.join()