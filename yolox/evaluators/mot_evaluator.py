from collections import defaultdict
import glob
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.bot_sort import BoTSORT
from yolox.tracker.damot_tracker import DAMOTTracker
from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker


from fast_reid.demo.inf_for_track import FeatureExtractionDemo, setup_cfg, postprocess_reid

import contextlib
import io
import os
import itertools
import json
import tempfile
import time


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_detection_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'

    # Convert results to the desired format
    formatted_results = []
    for d in results:
        frame_id = d['image_id']
        category_id = d['category_id']
        tlwhs = d['bbox']
        score = d['score']
        formatted_results.append((frame_id, category_id, tlwhs, score))

    with open(filename, 'w') as f:
        for frame_id, category_id, tlwh, score in formatted_results:
            x1, y1, w, h = tlwh
            line = save_format.format(frame=frame_id, id=category_id, x1=round(x1, 1), y1=round(y1, 1),
                                      w=round(w, 1), h=round(h, 1), s=round(score, 2))
            f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

        # self.resnet = fm.get_resnet18()
        # self.resnet = fm.get_resnet50()
        # self.resnet = fm.get_resnet152()
        # self.resnet = fm.get_mobilenet_v2()
        # self.resnet = fm.get_efficientnet_v2_l()
        # self.resnet = fm.get_AlexNet()

        cfg = setup_cfg(args)
        self.reid = FeatureExtractionDemo(cfg, parallel=args.parallel)


    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        save_data = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                # video_name for Visdrone2019
                if video_name == 'datasets':
                    video_name = img_file_name[0].split('/')[-2]

                # imgs = self.image_process(video_name, frame_id, imgs, info_imgs)

                self.args.track_buffer = 30
                self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                        ### Detection save
                        result_filename = os.path.join('YOLOX_outputs/damot_visdrone', 'det_{}.txt'.format(video_names[video_id - 1]))
                        write_detection_results(result_filename, save_data)
                        save_data = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)   # (1, 23625, 6)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre) # (n, 7)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids) # list( dict*n = {'image_id', 'category_id, 'bbox', 'score'} )
            data_list.extend(output_results)
            save_data.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                # object_features = self.extract_features(imgs, outputs[0][:, :4], info_imgs, tensor_type)
                online_targets = tracker.update(output_results=outputs[0], img_info=info_imgs, img_size=self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    # if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    if tlwh[2] * tlwh[3] > self.args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

                result_filename = os.path.join('YOLOX_outputs/damot_visdrone', 'det_{}.txt'.format(video_names[video_id]))
                write_detection_results(result_filename, save_data)


        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        boxes = np.asarray(boxes)
        if boxes.shape[0] == 0:
            return boxes
        boxes = np.copy(boxes)
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    def image_process(self, video_name, frame_id, image, info_imgs):
        test_dev_root = os.environ.get(
            'VISDRONE_TEST_DEV_ROOT',
            '/root/autodl-tmp/data/VisDrone2019-MOT-test-dev')
        gt_file = os.path.join(
            test_dev_root, 'annotations', video_name + '.txt')
        data = np.loadtxt(gt_file, delimiter=',')
        data = data[(data[:, 7] == 0) & (data[:, 0] == frame_id)]

        if data.size > 0:
            img_h, img_w = info_imgs[0].item(), info_imgs[1].item()
            x, y, w, h = data[0, 2:6]

            # Calculate scale factors
            height_scale = image.shape[2] / img_h
            width_scale = image.shape[3] / img_w

            # Scale bounding box coordinates
            x = int(x * width_scale)
            y = int(y * height_scale)
            w = int(w * width_scale)
            h = int(h * height_scale)

            # Ensure x, y, w, and h are within the image boundaries
            x = max(0, min(x, image.shape[2] - 1))
            y = max(0, min(y, image.shape[3] - 1))
            w = max(0, min(w, image.shape[2] - x))
            h = max(0, min(h, image.shape[3] - y))

            # Set the specified area to zero
            image[:, :, y:y + h, x:x + w] = 0

        return image

    def image_process_uavdt(self, video_name, frame_id, image, info_imgs):
        gt_file = os.path.join('datasets/uavdt/UAV-benchmark-MOTD_v1.0/GT', video_name + '_gt_ignore.txt')
        data = np.loadtxt(gt_file, delimiter=',')
        if data.size == 0:
            return image
        
        data = data[(data[:, 6] == 1) & (data[:, 0] == frame_id)]

        if data.size > 0:
            img_h, img_w = info_imgs[0].item(), info_imgs[1].item()
            x, y, w, h = data[0, 2:6]

            # Calculate scale factors
            height_scale = image.shape[2] / img_h
            width_scale = image.shape[3] / img_w

            # Scale bounding box coordinates
            x = int(x * width_scale)
            y = int(y * height_scale)
            w = int(w * width_scale)
            h = int(h * height_scale)

            # Ensure x, y, w, and h are within the image boundaries
            x = max(0, min(x, image.shape[2] - 1))
            y = max(0, min(y, image.shape[3] - 1))
            w = max(0, min(w, image.shape[2] - x))
            h = max(0, min(h, image.shape[3] - y))

            # Set the specified area to zero
            image[:, :, y:y + h, x:x + w] = 0

        return image

    def extract_features(self, original_bgr, bboxes, img_info):
        """Crop detections from the original frame and run standalone FastReID."""
        img_height = int(img_info[0].item())
        img_width = int(img_info[1].item())
        scale = min(
            self.img_size[0] / float(img_height),
            self.img_size[1] / float(img_width),
        )
        original_boxes = bboxes[:, :4].detach().cpu().numpy() / scale
        target_height, target_width = self.reid.cfg.INPUT.SIZE_TEST

        crops = []
        for x1, y1, x2, y2 in original_boxes:
            x1 = max(0, min(int(np.floor(x1)), img_width - 1))
            y1 = max(0, min(int(np.floor(y1)), img_height - 1))
            x2 = max(0, min(int(np.ceil(x2)), img_width))
            y2 = max(0, min(int(np.ceil(y2)), img_height))
            if x2 > x1 and y2 > y1:
                crop = original_bgr[y1:y2, x1:x2, ::-1]
                crop = cv2.resize(
                    crop,
                    (target_width, target_height),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                crop = np.zeros(
                    (target_height, target_width, 3), dtype=np.uint8
                )
            crops.append(
                torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))
                .float()
            )

        if not crops:
            return np.empty((0, 1), dtype=np.float32)

        reid_batch_size = max(int(getattr(self.args, "reid_batch_size", 128)), 1)
        feature_batches = []
        with torch.no_grad():
            for start in range(0, len(crops), reid_batch_size):
                batch = torch.stack(crops[start : start + reid_batch_size])
                output = self.reid.run_on_image_tensor(batch)
                feature_batches.append(postprocess_reid(output))

        features = np.concatenate(feature_batches, axis=0).reshape(len(crops), -1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.clip(norms, 1e-12, None)

    def evaluate_damot(
            self,
            model,
            distributed=False,
            half=False,
            trt_file=None,
            decoder=None,
            test_size=None,
            result_folder=None
    ):
        """Run DAMOT tracking and write one result file per video."""
        model = model.eval()
        if half:
            model = model.half()
        device = next(model.parameters()).device
        input_dtype = torch.float16 if half else torch.float32
        data_list = []
        results = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = max(len(self.dataloader), 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        tracker = DAMOTTracker(self.args)
        ori_thresh = self.args.track_thresh
        current_video_name = None

        def flush_results():
            if current_video_name is None:
                return
            result_filename = os.path.join(
                result_folder, "{}.txt".format(current_video_name)
            )
            write_results(result_filename, results)

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                frame_id = info_imgs[2].item()
                img_file_name = info_imgs[4]
                normalized_name = img_file_name[0].replace("\\", "/")
                video_name = normalized_name.rsplit("/", 2)[-2]

                self.args.track_buffer = 30
                self.args.track_thresh = ori_thresh

                if video_name != current_video_name:
                    flush_results()
                    results = []
                    tracker = DAMOTTracker(self.args)
                    current_video_name = video_name

                imgs = imgs.to(device=device, dtype=input_dtype, non_blocking=True)
                start = time.time()

                outputs = model(imgs)  # (1, 23625, 6)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)  # (n, 7)
                infer_end = time_synchronized()
                inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs,
                                                         ids)  # list( dict*n = {'image_id', 'category_id, 'bbox', 'score'} )
            data_list.extend(output_results)

            full_img_file_name = self.dataloader.dataset.resolve_image_path(
                img_file_name[0]
            )
            original_bgr = cv2.imread(full_img_file_name)
            if original_bgr is None:
                raise RuntimeError("Cannot read tracking frame: {}".format(full_img_file_name))

            if outputs[0] is None:
                tracker_detections = torch.empty((0, 7), device=device)
                object_features = np.empty((0, 1), dtype=np.float32)
            else:
                tracker_detections = outputs[0]
                object_features = self.extract_features(
                    original_bgr, tracker_detections[:, :4], info_imgs
                )

            online_targets = tracker.update(
                output_results=tracker_detections,
                img_info=info_imgs,
                id_feature=object_features,
                img_size=self.img_size,
                imgs=original_bgr,
            )
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for target in online_targets:
                tlwh = target.tlwh
                if tlwh[2] * tlwh[3] > self.args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(target.track_id)
                    online_scores.append(target.score)
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            track_end = time_synchronized()
            track_time += track_end - infer_end

        flush_results()

        statistics = torch.tensor(
            [inference_time, track_time, n_samples],
            dtype=torch.float32,
            device=device,
        )
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results
    
    def evaluate_sort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                # video_name for Visdrone2019
                if video_name == 'datasets':
                    video_name = img_file_name[0].split('/')[-2]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                # video_name for Visdrone2019
                if video_name == 'datasets':
                    video_name = img_file_name[0].split('/')[-2]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                # save results
                results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                # video_name for Visdrone2019
                if video_name == 'datasets':
                    video_name = img_file_name[0].split('/')[-2]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval
                logger.warning("Use standard COCOeval.")
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
