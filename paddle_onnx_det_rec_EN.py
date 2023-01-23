# https://blog.csdn.net/weixin_44898889/article/details/123827687
import sys
import cv2
import time
import math
import copy
import onnxruntime
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class NormalizeImage(object):

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale= eval(scale)
        self.scale= np.float32(scale if scale is not None else 1.0/ 255.0)
        mean= mean if mean is not None else [0.485, 0.456, 0.406]
        std= std if std is not None else [0.229, 0.224, 0.225]

        shape= (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean= np.array(mean).reshape(shape).astype('float32')
        self.std= np.array(std).reshape(shape).astype('float32')
    
    def __call__(self, data):
        img= data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img= np.array(img)

        assert isinstance(img,
            np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys
        
    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.limit_side_len = kwargs['limit_side_len']
        self.limit_type = kwargs.get('limit_type', 'min')

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type0(self, img):

        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            # print('11111', img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # return img, np.array([h, w])
        return img, [ratio_h, ratio_w]


class DBPostProcess(object): #
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                thresh =0.3,
                box_thresh =0.7,
                max_candidates =1000,
                unclip_ratio =2.0,
                use_dilation =False,
                **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)

        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        # print('points', contours)
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)

            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1=0
            index_4= 1
        else:
            index_1= 1
            index_4=0
        if points[3][1] > points[2][1]:
            index_2= 2
            index_3=3
        else:
            index_2=3
            index_3= 2
        
        box= [
            points[index_1], points[index_2], points[index_3], points[index_4]
            ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1) 

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        # print('segmentation', segmentation)
        boxes_batch = []
        for batch_index in range(pred.shape[0]):

            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                    src_w, src_h)
            
            boxes_batch.append({'points': boxes})
        return boxes_batch


#根据推理结果解码识别结果
class process_pred(object):
    def __init__(self, character_dict_path=None, character_type='en', use_space_char=True): #custom by default character_type='en', use_space_char=True
        self.character_str = ''
        with open(character_dict_path, 'rb') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip('\n').strip('\r\n')
                self.character_str += line
        if use_space_char:
            self.character_str += ' '
        dict_character = list(self.character_str)
        
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)

        preds_idx = preds.argmax(axis=2)

        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class det_rec_functions(object):

    def __init__(self, image, det_model, rec_model, en_dict):
        self.img = image.copy()
        self.det_file = det_model
        self.rec_file = rec_model
        self.rec_dict = en_dict
        self.onet_det_session = onnxruntime.InferenceSession(self.det_file)
        self.onet_rec_session = onnxruntime.InferenceSession(self.rec_file)
        self.infer_before_process_op, self.det_re_process_op= self.get_process()
        self.postprocess_op = process_pred(self.rec_dict, 'en', True)

    def transform(self, data, ops=None):
        """ transform """
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def create_operators(self, op_param_list, global_config =None):

        assert isinstance(op_param_list, list), ('operator config should be a list')
        ops = []
        for operator in op_param_list:
            assert isinstance(operator,
                dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            if global_config is not None:
                param.update(global_config)
            op = eval(op_name)(**param)
            ops.append(op)
        return ops

    # 检测框的后处理
    def order_points_clockwise(self, pts):

        xSorted = pts[np.argsort(pts[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            # print('字 符 检 测 框\n', box)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    # 定义图⽚前处理过程，和检测结果后处理过程
    def get_process(self):
        det_db_thresh = 0.3
        det_db_box_thresh = 0.5
        max_candidates = 2000
        unclip_ratio = 1.6
        use_dilation = True

        pre_process_list = [{
        'DetResizeForTest': {
        'limit_side_len': 2500,
        'limit_type': 'max'
        }
        }, {
        'NormalizeImage': {
        'std': [0.229, 0.224, 0.225],
        'mean': [0.485, 0.456, 0.406],
        'scale': '1./255.',
        'order': 'hwc'
        }
        }, {
        'ToCHWImage': None
        }, {
        'KeepKeys': {
        'keep_keys': ['image', 'shape']
        }
        }]

        infer_before_process_op = self.create_operators(pre_process_list)
        det_re_process_op = DBPostProcess(det_db_thresh, det_db_box_thresh, max_candidates, unclip_ratio, use_dilation)
        return infer_before_process_op, det_re_process_op

    def sorted_boxes(self, dt_boxes):

        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    # 图像输⼊预处理
    def resize_norm_img(self, img, max_wh_ratio):
        # imgC, imgH, imgW = [int(v) for v in "3, 32, 100".split(",")] # 根据识别模型修改尺⼨
        imgC, imgH, imgW = [int(v) for v in "3, 48, 320".split(",")] # 根据识别模型修改尺⼨
        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
        
    # 推理检测图⽚中的部分
    def get_boxes(self):
        img_ori = self.img
        img_part = img_ori.copy()
        data_part = {'image': img_part}
        data_part = self.transform(data_part, self.infer_before_process_op)
        img_part, shape_part_list = data_part
        img_part = np.expand_dims(img_part, axis=0)
        shape_part_list = np.expand_dims(shape_part_list, axis=0)

        inputs_part = {self.onet_det_session.get_inputs()[0].name: img_part}
        outs_part = self.onet_det_session.run(None, inputs_part)
        post_res_part = self.det_re_process_op(outs_part[0], shape_part_list)
        
        dt_boxes_part = post_res_part[0]['points']
        dt_boxes_part = self.filter_tag_det_res(dt_boxes_part, img_ori.shape)
        dt_boxes_part = self.sorted_boxes(dt_boxes_part)
        return dt_boxes_part

    # 根据bounding box 得到单元格图⽚
    def get_rotate_crop_image(self, img, points):

        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)

        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)

        return dst_img

    # 单张图⽚推理
    def get_img_res(self, onnx_model, img, process_op):

        h, w = img.shape[:2]
        img = self.resize_norm_img(img, w * 1.0 / h)
        img = img[np.newaxis, :]
        inputs = {onnx_model.get_inputs()[0].name: img}
        outs = onnx_model.run(None, inputs)
        result = process_op(outs[0])
        return result

    def recognition_img(self, dt_boxes):
        img_ori = self.img
        img = img_ori.copy()
        # 识别过程
        # 根据bndbox 得到剪裁后的图⽚
        img_list = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)

            img_crop = self.get_rotate_crop_image(img, tmp_box)
            img_crop = cv2.resize(img_crop, (100, 32))
            img_list.append(img_crop)
            
        results = []
        results_info = []
        for pic in img_list:
            cv2.imwrite('../rec_img/rec.jpg', pic)
            # cv2.waitKey(0)
            res = self.get_img_res(self.onet_rec_session, pic, self.postprocess_op)
            results.append(res[0])
            results_info.append(res)
        return results, results_info

    def recognition_pil(self, img_list):
        img_list = [cv2.resize(np.array(el), (320, 48)) for el in img_list] #custom changed img size
        results = []
        results_info = []
        for pic in img_list:
            cv2.imwrite('../rec_img/rec.jpg', pic)
            res = self.get_img_res(self.onet_rec_session, pic, self.postprocess_op)
            results.append(res[0])
            results_info.append(res)
        return results, results_info
# if __name__ == '__main__':
#     #读取图⽚
#     det_model = r'det_server.onnx' # 检测模型路径
#     rec_model = r'dynamic_rec.onnx' # 识别模型路径
#     image = cv2.imread('1940422.jpg') # 填⼊待检测识别的图⽚
#     # image = cv2.resize(image, (320, 320)) # 根据onnx 模型输⼊尺⼨进⾏调整
#     # OCR检测-识别
#     ocr_sys = det_rec_functions(image)
#     #得到检测框
#     starttime = time.time()
#     dt_boxes = ocr_sys.get_boxes()
#     # print(dt_boxes)
#     # 识别 results:单纯的识别结果，results_info:识别结果 + 置信度
#     results, results_info = ocr_sys.recognition_img(dt_boxes)
#     endtime = time.time()
#     results = results[0][0].replace(' ', '') #去除空格
#     print('识别结果: ', results)
#     print("predittime: {}".format(endtime - starttime))