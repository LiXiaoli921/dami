# encoding=utf-8
"""
Copyright (c) 2018 Heils.cn
Written by kohill.
"""
from __future__ import print_function
from __future__ import division
import sys
# reload(sys)
sys.setdefaultencoding('utf-8')
import logging
import time
import os
import glob
import cv2
import threading
# import tqdm
# Do not import any DL framework here, including tensorflow, torch, etc.
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue
import numpy as np
# from easydict import EasyDict


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


# def worker_image_pre_process(da_dict, g_cfg, q_in, q_formula_detection):
#     # type: (dict, EasyDict, Queue, Queue) -> None
#     from flip_classify.flip import flip_init, flip_classify
#
#     flip_model, flip_transform = flip_init(g_cfg.resume, g_cfg.gpu_id)
#     try:
#
#         while True:
#             im_id = q_in.get()
#             if im_id is None:
#                 q_formula_detection.put(None)
#                 logging.warning("worker_image_pre_process is existing.")
#                 return None
#             img = da_dict[im_id]["data"]
#             img = np.asarray(img)
#             if len(img.shape) == 2:
#                 print('single channel')
#                 img = cv2.merge([img, img, img])
#             if len(img.shape) != 3:
#                 img = np.zeros([50,100,3],np.uint8)
#                 logger.info('error_image_shape')
#             img = img[:, :, :3]
#             height, width, chanel = img.shape
#             if width > g_cfg.new_width:
#                 new_width = g_cfg.new_width
#                 new_height = int(new_width * height * 1.0 / width)
#                 img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)  # 双三次差值
#             flipped = flip_classify(img, flip_model, flip_transform, g_cfg.gpu_id)
#
#             height, width, chanel = flipped.shape
#             if width > g_cfg.new_width:
#                 new_width = g_cfg.new_width
#                 new_height = int(new_width * height * 1.0 / width)
#                 flipped = cv2.resize(flipped, (new_width, new_height), interpolation=cv2.INTER_AREA)  # 双三次差值
#             # Write data back.
#             im_da = da_dict[im_id]
#             im_da["data"] = flipped
#             data_dict[im_id] = im_da
#             q_formula_detection.put(im_id)  # Tell formula detector this image has been pre-processed
#     except Exception as e:
#         logging.exception(e)
#         logger.info(e)
#
# def worker_formula_detector(da_dict, g_cfg, q_in, queue_text_detector):
#     # type: (dict, EasyDict, Queue, Queue) -> None
#     # from formular_dect.main import formula_detector_init, formula_detect
#     from formular_detection_v2.demo import formula_detector_init, formula_detect
#     gpu_id = g_cfg.gpu_id
#     logging.info("Initializing formula detector.")
#     # net, nms_wrapper, ctx = formula_detector_init(gpu_id)
#     ctx, config, net = formula_detector_init(gpu_id)
#
#     logging.info("Formula detector initializing finished.")
#     try:
#         while True:
#             im_id = q_in.get(block=True)
#             if im_id is None:
#                 queue_text_detector.put(None)
#                 logging.warning("worker_formula_detector is existing.")
#                 break
#             im = da_dict[im_id]["data"]
#             # masked, formulars, formular_box, formular_box_ori, tables, table_box, table_box_ori, pics, pic_box, pic_box_ori, table_item = formula_detect(
#             #     net, nms_wrapper, im, ctx)
#             masked, formulars, formular_box, formular_box_ori, tables, table_box, table_box_ori, pics, pic_box, pic_box_ori, table_item = formula_detect(ctx,net,config,im)
#
#             dict_pic = {}
#             index = 0
#             for i in pic_box_ori:
#                 i += [3]
#                 key = str(i)[1:-1]
#                 dict_pic[key] = pics[index]
#
#                 index += 1
#
#             im_da = da_dict[im_id]
#             im_da['dict_pic'] = dict_pic
#             # im_da['pics'] = pics
#             im_da["masked"] = masked  # Masked image for text detector
#             im_da["formula_cropped_images"] = formulars  # Cropped Images for formula recognition.
#             im_da["formula_boxes"] = formular_box_ori  # Formula bounding boxes. [0,0,1,0,0,1,1,1]
#             im_da["formula_boxes_shrunk"] = formular_box  # Formula bounding boxes which are shrunk (1 over 3).
#             im_da['table_box_ori'] = table_box_ori  # table box  [0,0,1,0,0,1,1,1]
#             im_da['pic_box_ori'] = pic_box_ori  # pices box   [0,0,1,0,0,1,1,1]
#             print(table_item)
#             im_da['table_item'] = table_item  # table item box    [0,0,1,1]
#             da_dict[im_id] = im_da
#
#             queue_text_detector.put(im_id)  # Tell Text detector this image has been processed by formula detector
#     except Exception as e:
#         logging.exception(e)
#         print(e)
#
# def worker_text_detector(da_dict, g_cfg, q_in, queue_construct_text_lines):
#     # type: (dict, EasyDict, Queue, Queue) -> None
#     # from Detection.detector_v6_pb import *
#     from Detection.detector_v9_pytorch import Detector
#     logging.info("Initializing Text detector.")
#     # detector = Detector(g_cfg.path_yml, g_cfg.path_ckpt, g_cfg.gpu_id)  # 文本检测
#     detector = Detector(g_cfg.path_cfg_anchor_list, g_cfg.path_ckpt, g_cfg.gpu_id)
#     logging.info("Text detector initializing finished.")
#     try:
#         while True:
#             im_id = q_in.get(block=True)
#             if im_id is None:
#                 queue_construct_text_lines.put(None)
#                 logging.warning("Worker_text_detector is existing.")
#                 break
#             im_da = da_dict[im_id]
#             masked = im_da["masked"]
#             # img = im_da['data']
#
#             # text_boxes = detector.detect_text(masked)
#
#             start = time.time()
#             # text_box = detector.detect_text(masked)
#             # boxes1, scores1, img1, pad_para1, scale1, ori_w1, ori_h1 = detector.detect_text(masked, False)
#             cls_prob1, bbox1, pad_para1, scale1, ori_w1, ori_h1 = detector.detect_text(masked)
#             # print('1 step time cost: {} ms'.format((time.time() - start) * 1000))
#
#             im_da['cls_prob1'] = cls_prob1
#             im_da['bbox1'] = bbox1
#             im_da['pad_para1'] = pad_para1
#             im_da['scale1'] = scale1
#             im_da["ori_w1"] = ori_w1
#             im_da["ori_h1"] = ori_h1
#             im_da['nms_gpu_id'] = g_cfg.gpu_id
#
#             da_dict[im_id] = im_da
#
#             queue_construct_text_lines.put(im_id)  # Tell Text recognizer this image has been processed by formula detector
#     except Exception as e:
#         logging.exception(e)
#         print(e)
#
# def worker_construct_text_lines(da_dict, q_in, queue_text_recognizer):
#     # type: (dict, EasyDict, Queue, Queue) -> None
#     # from Detection.detector_v6_pb import rotation_and_crop, construct_text_lines
#     from Detection.detector_v9_pytorch import rotation_and_crop, construct_text_lines
#     from cut_item_fn import cut_boxes
#     from merge_box import change_context
#     from sort_boxes import get_character
#     logging.info(" construct_text_lines init finished.")
#     try:
#         while True:
#             im_id = q_in.get(block=True)
#             if im_id is None:
#                 queue_text_recognizer.put(None)
#                 logging.warning("Worker_text_detector is existing.")
#                 break
#             im_da = da_dict[im_id]
#
#             masked = im_da["masked"]
#             # print('############')
#             # print(masked)
#             # cv2.imshow('2asd',masked)
#             # cv2.waitKey(0)
#             # time.sleep(15)
#             cls_prob1 = im_da['cls_prob1']
#             bbox1 = im_da['bbox1']
#             pad_para1 = im_da['pad_para1']
#             scale1 = im_da['scale1']
#             ori_w1 = im_da["ori_w1"]
#             ori_h1 = im_da["ori_h1"]
#             nms_gpu_id = im_da['nms_gpu_id']
#
#             table_item = im_da["table_item"]
#             table_box_ori = im_da["table_box_ori"]
#             formular_box_ori = im_da['formula_boxes']
#             formular_box = im_da['formula_boxes_shrunk']
#             # text_boxes = construct_text_lines(boxes1, scores1, img1, pad_para1, scale1, ori_w1, ori_h1)
#             text_boxes = construct_text_lines(cls_prob1, bbox1, pad_para1, scale1, ori_w1, ori_h1, nms_gpu_id)
#
#             start_t = time.time()
#             print(table_item)
#             if len(table_item) > 0 and len(table_box_ori) and len(text_boxes)> 0:
#                 table_item, text_boxes = cut_boxes(np.asarray(table_item), np.asarray(text_boxes), th=0.05)
#             print(table_item)
#
#             # print('2 step time cost: {} ms'.format((time.time() - start) * 1000))
#
#             # im_da["text_boxes"] = text_boxes
#
#             final_dict = change_context(masked.copy(), text_boxes, formular_box_ori)
#             # print('     change context time: {}ms'.format((time.time() - start) * 1000))
#             # start = time.time()
#             formula_small_dict = {str(a)[1:-1]: str(b)[1:-1] for a, b in zip(formular_box, formular_box_ori)}
#             formula_big_dict = {str(b)[1:-1]: str(a)[1:-1] for a, b in zip(formular_box, formular_box_ori)}
#
#             text_box_split, all_box = get_character(final_dict, formula_big_dict)
#             masked = np.asarray(masked)
#
#             im_da['text_boxes']   = text_boxes
#             im_da['text_box_split'] = text_box_split
#
#             im_da["all_box"] = all_box
#             im_da["text_boxes_images_cropped_rotated"] = [np.asarray(rotation_and_crop(masked, x[:-1])) for x in
#                                                          text_box_split]
#             # im_da["text_boxes_images_cropped_rotated"] =map(rotation_and_crop,masked,text_box_split)
#             # print('     rotate time: {}ms'.format((time.time() - start) * 1000))
#             da_dict[im_id] = im_da
#             # print('     construct time: {}ms'.format((time.time() - start_0) * 1000))
#
#             queue_text_recognizer.put(im_id)  # Tell Text recognizer this image has been processed by formula detector
#             # print('[detect_text] spent {0:f} ms'.format(time.time() - start))
#     except Exception as e:
#         logging.exception(e)
#         print(e)
#
# def worker_text_recognizer(da_dict, g_cfg, q_in, q_out):
#     # type: (dict, EasyDict, Queue, Queue) -> None
#     from Recognization.img_long_reference import net_recog, tensor_to_result
#     # from formular_rec.demo_release import Predictor
#     from release_kohill.demo_release import Predictor
#     import json
#     from xml2table_2box_fn import filter_pic
#     from sort_boxes import final_sort
#     import re
#     from scripts.bracket_match import check
#     from scripts.bracket_match_v2 import check_v2
#     rec_net = Predictor(gpu_id=g_cfg.gpu_id, decoder_max_len=50)  # 公式识别
#     model, index_word_dict = net_recog(g_cfg.gpu_id)  # 文本识别
#     try:
#         while True:
#             start = time.time()
#             im_id = q_in.get(block=True)
#             if im_id is None:
#                 q_out.put(None)
#                 logging.warning("worker_text_recognizer is existing.")
#                 break
#             im_da = da_dict[im_id]
#             formula_results_dict = {}
#             lhl_formula_results_dict = {}
#             def _worker(r,w):
#             # type: (dict) -> None
#                 # Formula recognition
#                 formula_results = rec_net(im_da["formula_cropped_images"])
#                 # formula_results = [[] for x in im_da["formula_cropped_images"]]
#                 # print(im_da["formula_results"])
#                 index2 = 0
#                 tmp_formula_results = []
#                 tmp_formula_boxes_shrunk = []
#                 tmp_formular_formula_boxes =[]
#                 for item1 in formula_results:
#                     try:
#                         item = check_v2(item1)
#                     except:
#                         item = check(item1)
#                     if item != item1:
#
#                         logger.info('formular old: {}'.format(item1))
#                         logger.info('formular new: {}'.format(item))
#                     if re.match(r"^[a-zA-Z.()0-9,\"|\-']*$", item):
#                         print('english item {}'.format(item))
#                         im_da['text_box_split'].append(im_da["formula_boxes"][index2]+[1])
#                         im_da["all_box"].append(im_da["formula_boxes"][index2]+[1])
#                         formular_pic_tmp =im_da["formula_cropped_images"][index2]
#                         im_da["text_boxes_images_cropped_rotated"].append(formular_pic_tmp)
#                     else:
#                         tmp_formula_results.append(formula_results[index2])
#                         tmp_formula_boxes_shrunk.append(im_da["formula_boxes_shrunk"][index2])
#                         tmp_formular_formula_boxes.append(im_da["formula_boxes"][index2])
#                     index2 +=1
#
#                 im_da["formula_boxes_shrunk"] = tmp_formula_boxes_shrunk
#                 im_da["formula_boxes"] = tmp_formular_formula_boxes
#                 formula_results = tmp_formula_results
#                 formula_results_dict_l = {json.dumps(b)[1:-1]: "".join(r) for b, r in
#                                         zip(im_da["formula_boxes_shrunk"], formula_results)}
#
#                 formula_results_dict_lhl = {json.dumps(b)[1:-1]: "".join(r) for b, r in
#                                       zip(im_da["formula_boxes"], formula_results)}
#
#                 r.update(formula_results_dict_l)
#                 w.update(formula_results_dict_lhl)
#                 # im_da["formula_results"] = formula_results_dict
#                 da_dict[im_id] = im_da
#
#
#             _worker(formula_results_dict,lhl_formula_results_dict)
#
#             # t_formula_recognition = threading.Thread(target=_worker, args=(formula_results_dict, ))
#             # t_formula_recognition.daemon = True
#             # t_formula_recognition.start()
#
#             # Text recognition
#             im_da = da_dict[im_id]
#             text_boxes_images_cropped_rotated = im_da["text_boxes_images_cropped_rotated"]
#             text_box_split = im_da['text_box_split']
#             all_box = im_da["all_box"]
#             # pics = im_da['pics']
#             dict_pic = im_da['dict_pic']
#             character_results = []
#
#
#             for i in range(len(text_boxes_images_cropped_rotated)):
#                 if text_boxes_images_cropped_rotated[i].size <= 1:
#                     try:
#                         cv2.imwrite('bugpic/bug{}.png'.format(time.time()), im_da['data'])
#                     except Exception as e:
#                         logging.exception(e)
#                     logging.warning("Find illegal cropped image.")
#                     logger.info('Find illegal cropped image.’')
#                     text_boxes_images_cropped_rotated[i] = np.zeros(shape=(2, 2, 3), dtype=np.uint8)
#             for i in range(len(text_boxes_images_cropped_rotated) // 12 + 1):
#                 x = text_boxes_images_cropped_rotated[i * 12:(i + 1) * 12]
#                 if len(x):
#                     try:
#                         r = tensor_to_result(x, model, index_word_dict, g_cfg.gpu_id)
#                     except:
#                         r = ['error']*len(r)
#                     character_results.extend(r)
#             text_rotate_pic_dict = {b: r for b, r in zip(character_results, text_boxes_images_cropped_rotated)}
#             # t_formula_recognition.join()
#             character_results_dict = {json.dumps(b)[1:-1]: r for b, r in zip(text_box_split, character_results)}
#
#             table_box_ori = im_da['table_box_ori']
#             pic_box_ori = im_da['pic_box_ori']
#
#             table_dict = {}
#             if len(table_box_ori+pic_box_ori) >0:
#                 all_box = filter_pic(all_box, table_box_ori+pic_box_ori)
#             for i in table_box_ori:
#                 i += [2]
#                 all_box.append(i)
#             for i in pic_box_ori:
#                 i += [3]
#                 all_box.append(i)
#             character_boxs,box_number = final_sort(all_box)
#
#             obj = {}
#             obj['rect'] = []
#             context = []
#
#             from xml2tabl_col_row_fn import boxes2table
#             # print(len(text_box_split))
#             table_item = im_da['table_item']
#             table_box_ori = im_da['table_box_ori']
#             html_content = ''
#             forular_str_box_html = lhl_formula_results_dict.keys()
#             forular_result_box_html = [lhl_formula_results_dict[item] for item in forular_str_box_html]
#             forular_box_html = [map(int, item.split(',')) for item in forular_str_box_html]
#             im_da['xml_data'] = []
#             im = im_da['data']
#             im_da['ceshi_tmp'] = im
#             # print('^^^')
#             # print(table_item)
#             # print(table_box_ori)
#             print(im_da['text_box_split'])
#             if len(table_item) > 0 and len(table_box_ori) > 0 and len(im_da['text_box_split']) > 0:
#                 # print('^^^')
#                 try:
#                     # for a in character_results:
#                     #     print(a)
#
#                     for a in table_box_ori:
#
#                         # for box in forular_box_html:
#                         #     #
#                         #     cv2.line(im, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), (255, 255, 0), 1)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[0]), int(box[3])), (255, 255, 0), 1)
#                         #     cv2.line(im, (int(box[2]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[0]), int(box[1])), (255, 255, 0), 1)
#                         #
#                         # for box in table_box_ori:
#                         #     #
#                         #     cv2.line(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[6]), int(box[7])), (255, 0, 0), 1)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), (255, 0, 0), 1)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[0]), int(box[1])), (255, 0, 0), 1)
#
#                         # for box in text_box_split:
#                         #     #
#                         #     cv2.line(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[6]), int(box[7])), (0, 255, 255), 2)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), (0, 255, 255), 2)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[0]), int(box[1])), (0, 255, 255), 2)
#                         # for box in forular_box_html:
#                         #     #
#                         #     cv2.line(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[6]), int(box[7])), (0, 0, 255), 2)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), (0, 0, 255), 2)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[0]), int(box[1])), (0, 0, 255), 2)
#                         # for box in table_item:
#                         #     #
#                         #     cv2.line(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[1])), (128, 0, 128), 1)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[2]), int(box[1])), (128, 0, 128), 1)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[0]), int(box[3])), (128, 0, 128), 1)
#                         #     cv2.line(im, (int(box[0]), int(box[1])), (int(box[0]), int(box[3])), (128, 0, 128), 1)
#
#                         from sort_boxes import sort4table
#                         box4table, context4table = sort4table(
#                             [x[:8] for x in text_box_split] + [x[:8] for x in forular_box_html],
#                             character_results + forular_result_box_html)
#                         character_boxes = []
#                         index1 = 0
#                         # for box in box4table:
#                         #     #
#                         #     cv2.line(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
#                         #     cv2.line(im, (int(box[2]), int(box[3])), (int(box[6]), int(box[7])), (0, 0, 255), 2)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), (0, 0, 255), 2)
#                         #     cv2.line(im, (int(box[4]), int(box[5])), (int(box[0]), int(box[1])), (0, 0, 255), 2)
#                         # cv2.imshow('ori', im)
#                         # cv2.waitKey(0)
#                         for x in box4table:
#                             character_boxes.append(x + [index1])
#                             index1 += 1
#                         im_da['xml_data'] = [box4table, context4table]
#                         html_content_tmp = boxes2table(np.asarray([a]), np.asarray(table_item),
#                                                        np.asarray(character_boxes), context4table)
#
#                         # html_content += html_content_tmp
#
#                         key = str(a[:-1])[1:-1]
#                         # print('***')
#                         # print(key)
#                         table_dict[key] = html_content_tmp
#
#                     # print(html_content)
#                     # print(im_da['name'])
#                     # cv2.imshow('ori', im)
#                     # cv2.waitKey(0)
#
#                     im_da['ceshi_tmp'] = im
#                     # with open('{}.html'.format(im_da['name']),'w') as f:
#                     #     f.write(html_content)
#                     # cv2.imshow('',im_da['data'])
#                     # cv2.waitKey(0)
#                     # cv2.destroyAllWindows()
#                 except Exception as e:
#                     print('html_error')
#                     print(e)
#                 # print(html_content)
#                 # print('[html] spent {0:f} s'.format(time.time() - start))
#
#
#
#             countbox = 0
#             case = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
#             for box in character_boxs:
#                 num = len(box_number)
#                 if len(box) == 9:
#                     if box[8] == 0:
#                         # logging.debug(box)
#                         # logging.debug(im_da["formula_results"])
#                         tmp = str(box[:-1])[1:-1]
#                         # tmp = formula_small_dict[tmp]
#                         # box = tmp.split(',')
#                         # formular = formula_results_dict[tmp]
#                         # formular = im_da["formula_results"][tmp]
#                         try:
#                             formular = formula_results_dict[tmp]
#
#                             context.append('$$'+formular+'$$')
#                             # print(formular.encode('utf-8'))
#                             # rect2xml = ['###' + formular.encode('utf-8'), str(int(box[0])) + ',' + str(int(box[1])),
#                             #             str(int(box[2])) + ',' + str(int(box[3])),
#                             #             str(int(box[4])) + ',' + str(int(box[5])),
#                             #             str(int(box[6])) + ',' + str(int(box[7]))]
#                             #
#                             # obj['rect'].append(rect2xml)
#                         except:
#                             print('delete')
#                     elif box[8] == 1:
#                         tmp = str(box)[1:-1]
#                         # logging.debug('***recog')
#                         # logging.debug(tmp)
#                         # logging.debug(character_results_dict)
#                         # print('imagename{}'.format(im_da['name']))
#                         character = character_results_dict[tmp]
#                         try:
#                             if len(context) > 0 and character[0] in case and context[-1][-1] in case:
#                                 context.append(' ' + character)
#                             else:
#                                 context.append(character)
#                         except:
#                             context.append(character)
#
#                         # context.append(character)
#                         # print(character)
#                         # print(character)
#                         # rect2xml = ['1111' + character, str(int(box[0])) + ',' + str(int(box[1])),
#                         #             str(int(box[2])) + ',' + str(int(box[3])),
#                         #             str(int(box[4])) + ',' + str(int(box[5])),
#                         #             str(int(box[6])) + ',' + str(int(box[7]))]
#                         # obj['rect'].append(rect2xml)
#                     elif box[8] == 2:
#                         # table
#                         # print(box[8])
#                         # print(type(box))
#                         # print(str(box)[1:-1])
#                         # print('table')
#                         # print(table_dict[str(box)[1:-1]])
#                         context.append(table_dict[str(box)[1:-1]])
#
#                     elif box[8] == 3:
#                         # pic
#                         # print('pic')
#                         # print(dict_pic[str(box)[1:-1]])
#                         context.append(dict_pic[str(box)[1:-1]])
#
#                 countbox += 1
#
#                 if countbox == box_number[0]:
#                     context.append('\n')
#                     try:
#                         box_number = box_number[1:]
#                         countbox = 0
#                     except:
#                         logging.debug('final')
#
#             # context.append('\n\n')
#             # context.append(html_content.encode('utf-8'))
#             # print(html_content)
#             # context.append('\n\n')
#             # for i in pics:
#             #     context.append(i.encode('utf-8'))
#             #     print(i)
#             #     print([i])
#             #     context.append('\n\n')
#             # print(context)
#
#             im_da["character_results"] = character_results
#             im_da["context"] = context
#
#
#             im_da['html_content'] = html_content
#             logging.info(context)
#             da_dict[im_id] = im_da
#             q_out.put(im_id)
#             # print('[recog text] spent {0:f} s'.format(time.time() - start))
#     except Exception as e:
#         logging.exception(e)
#         print(e)

im_count_as_id = 0

types_count_hist = None


def get_gc_stats():
    import gc
    gc.collect(generation=2)
    objects = gc.get_objects()
    object_types = [type(x) for x in objects]
    types = list(set(object_types))
    types_count = {x: object_types.count(x) for x in types}
    global types_count_hist
    if types_count_hist is not None:
        for k in types_count:
            if k in types_count_hist:
                if types_count[k] - types_count_hist[k] > 1 and types_count[k] > 10:
                    print (k, types_count[k])
    types_count_hist = types_count
    # print (types[int(np.argmax(types_count))], np.max(types_count))


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
    from cStringIO import StringIO as StringIO
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