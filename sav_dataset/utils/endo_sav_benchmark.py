# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the sav_dataset directory of this source tree.

# adapted from https://github.com/hkchengrex/vos-benchmark
# and  https://github.com/davisvideochallenge/davis2017-evaluation
# with their licenses found in the LICENSE_VOS_BENCHMARK and LICENSE_DAVIS files
# in the sav_dataset directory.
import math
import os
import time
import json
from collections import defaultdict
from multiprocessing import Pool
from os import path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image
from skimage.morphology import disk

class VideoEvaluator:
    def __init__(self, gt_root, pred_root, skip_first_and_last=True) -> None:
        """
        gt_root: path to the folder storing the gt masks
        pred_root: path to the folder storing the predicted masks
        skip_first_and_last: whether we should skip the evaluation of the first and the last frame.
                             True for SA-V val and test, same as in DAVIS semi-supervised evaluation.
        """
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.skip_first_and_last = skip_first_and_last

    def __call__(self, data) -> Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        vid_name: name of the video to evaluate
        """
        # scan the folder to find subfolders for evaluation and
        # check if the folder structure is SA-V
        # vid_name, first_frame_ids, object_ids = data
        vid_name, evaluate_object_ids = data
        to_evaluate, is_sav_format = self.scan_vid_folder(vid_name, evaluate_object_ids)

        eval_results = []
        for all_frames, obj_id, gt_path, pred_path in to_evaluate:
            obj_id = int(obj_id)
            if self.skip_first_and_last:
                # skip the first and the last frames
                all_frames = all_frames[1:-1]

            evaluator = Evaluator(name=vid_name, obj_id=obj_id)
            for frame in all_frames:
                gt_array, pred_array = self.get_gt_and_pred(
                    gt_path, pred_path, frame, is_sav_format, object_id=obj_id
                )
                evaluator.feed_frame(mask=pred_array, gt=gt_array, frame=frame, object_id=obj_id)
            iou, boundary_f, dice = evaluator.conclude()
            eval_results.append((obj_id, iou, boundary_f, dice))

        if is_sav_format:
            iou_output, boundary_f_output, dice_output = self.consolidate(eval_results)
        else:
            assert len(eval_results) == 1
            iou_output = eval_results[0][1]
            boundary_f_output = eval_results[0][2]
            dice_output = eval_results[0][3]

        return vid_name, iou_output, boundary_f_output, dice_output

    def get_gt_and_pred(
        self,
        gt_path: str,
        pred_path: str,
        f_name: str,
        is_sav_format: bool,
        object_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the ground-truth and predicted masks for a single frame.
        """
        gt_mask_path = path.join(gt_path, f_name)
        pred_mask_path = path.join(pred_path, f_name)
        assert os.path.exists(pred_mask_path), f"{pred_mask_path} not found"

        gt_array = np.array(Image.open(gt_mask_path))
        pred_array = np.array(Image.open(pred_mask_path))
        assert (
            gt_array.shape[-2:] == pred_array.shape[-2:]
        ), f"shape mismatch: {gt_mask_path}, {pred_mask_path}"

        if is_sav_format:
            assert len(np.unique(pred_array)) <= 2, (
                f"found more than 1 object in {pred_mask_path} "
                "SA-V format assumes one object mask per png file."
            )
            gt_array = gt_array == object_id
            assert len(np.unique(gt_array)) <= 2, (
                f"found more than 1 object in {gt_mask_path} "
                "SA-V format assumes one object mask per png file."
            )
            pred_array = pred_array == object_id

        return gt_array, pred_array

    def scan_vid_folder(self, vid_name, evaluate_object_ids) -> Tuple[List, bool]:
        """
        Scan the folder structure of the video and return a list of folders for evaluate.
        """
        vid_gt_path = path.join(self.gt_root, vid_name)
        vid_pred_path = path.join(self.pred_root, vid_name)
        all_files_and_dirs = sorted(os.listdir(vid_pred_path))
        to_evaluate = []
        is_sav_format = True
        for obj_dir in all_files_and_dirs:
            # check if the object is a integer
            if (not obj_dir.isdigit()
                    or not path.isdir(path.join(vid_pred_path, obj_dir))
                    or int(obj_dir) not in evaluate_object_ids  # skip the object not in the list
            ):
                continue
            obj_gt_path = vid_gt_path
            obj_pred_path = path.join(vid_pred_path, obj_dir)
            frames = sorted(os.listdir(obj_pred_path))
            # check if the frame is in the gt folder
            for frame in frames:
                if not os.path.exists(path.join(obj_gt_path, frame)):
                    print(f"frame {frame} not found in {obj_gt_path}")
                    frames.remove(frame)
            to_evaluate.append((frames, obj_dir, obj_gt_path, obj_pred_path))
        return to_evaluate, is_sav_format

    @staticmethod
    def consolidate(
        eval_results
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        Consolidate the results of all the objects from the video into one dictionary.
        """
        iou_output = {}
        boundary_f_output = {}
        dice_output = {}
        for obj_id, iou, boundary_f, dice in eval_results:
            assert len(iou) == 1
            key = list(iou.keys())[0]
            iou_output[obj_id] = iou[key]
            boundary_f_output[obj_id] = boundary_f[key]
            dice_output[obj_id] = dice[key]
        return iou_output, boundary_f_output, dice_output


#################################################################################################################
# Functions below are from https://github.com/hkchengrex/vos-benchmark with minor modifications
# _seg2bmap from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/utils.py
# get_iou and Evaluator from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/evaluator.py
# benchmark from https://github.com/hkchengrex/vos-benchmark/blob/main/vos_benchmark/benchmark.py with slight mod
#################################################################################################################


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def get_iou(intersection, pixel_sum):
    # handle edge cases without resorting to epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        assert intersection == 0
        return 1

    return intersection / (pixel_sum - intersection)

def get_dice(intersection, pixel_sum):
    # handle edge cases without resorting to epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        assert intersection == 0
        return 1

    return 2 * intersection / pixel_sum


class Evaluator:
    def __init__(self, boundary=0.008, name=None, obj_id=None):
        # boundary: used in computing boundary F-score
        self.boundary = boundary
        self.name = name
        self.obj_id = obj_id
        self.objects_in_gt = set()
        self.objects_in_masks = set()

        self.object_iou = defaultdict(list)
        self.frame_fg_object_iou = []
        self.object_dice = defaultdict(list)
        self.boundary_f = defaultdict(list)

    def feed_frame(self, mask: np.ndarray, gt: np.ndarray, frame, object_id=0):
        """
        Compute and accumulate metrics for a single frame (mask/gt pair)
        """

        # get all objects in the ground-truth
        gt_objects = np.unique(gt)
        gt_objects = gt_objects[gt_objects != 0].tolist()

        # get all objects in the predicted mask
        mask_objects = np.unique(mask)
        mask_objects = mask_objects[mask_objects != 0].tolist()

        self.objects_in_gt.update(set(gt_objects))
        self.objects_in_masks.update(set(mask_objects))

        all_objects = self.objects_in_gt.union(self.objects_in_masks)  #

        # boundary disk for boundary F-score. It is the same for all objects.
        bound_pix = np.ceil(self.boundary * np.linalg.norm(mask.shape))
        boundary_disk = disk(bound_pix)

        for obj_idx in all_objects:
            obj_mask = mask == obj_idx
            obj_gt = gt == obj_idx

            iou_value = get_iou((obj_mask * obj_gt).sum(), obj_mask.sum() + obj_gt.sum())
            # For challenge IoU, we only consider the object in the ground-truth
            if np.sum(obj_gt) != 0:
                self.frame_fg_object_iou.append((frame, object_id, iou_value))
            # object iou
            self.object_iou[obj_idx].append(
                iou_value
            )
            self.object_dice[obj_idx].append(
                get_dice((obj_mask * obj_gt).sum(), obj_mask.sum() + obj_gt.sum())
            )
            """
            # boundary f-score
            This part is copied from davis2017-evaluation
            """
            mask_boundary = _seg2bmap(obj_mask)
            gt_boundary = _seg2bmap(obj_gt)
            mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
            gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

            # Get the intersection
            gt_match = gt_boundary * mask_dilated
            fg_match = mask_boundary * gt_dilated

            # Area of the intersection
            n_fg = np.sum(mask_boundary)
            n_gt = np.sum(gt_boundary)

            # Compute precision and recall
            if n_fg == 0 and n_gt > 0:
                precision = 1
                recall = 0
            elif n_fg > 0 and n_gt == 0:
                precision = 0
                recall = 1
            elif n_fg == 0 and n_gt == 0:
                precision = 1
                recall = 1
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

            # Compute F measure
            if precision + recall == 0:
                F = 0
            else:
                F = 2 * precision * recall / (precision + recall)
            self.boundary_f[obj_idx].append(F)

    def conclude(self):
        all_iou = {}
        all_dice = {}
        all_boundary_f = {}
        for object_id in self.objects_in_gt:
            all_iou[object_id] = np.mean(self.object_iou[object_id]) * 100
            all_boundary_f[object_id] = np.mean(self.boundary_f[object_id]) * 100
            all_dice[object_id] = np.mean(self.object_dice[object_id]) * 100

        return all_iou, all_boundary_f, all_dice


def benchmark(
    gt_roots,
    mask_roots,
    strict=True,
    num_processes=None,
    *,
    verbose=True,
    skip_first_and_last=True,
    epoch=0,
    evaluate_object_id_list=None,
):
    """
    gt_roots: a list of paths to datasets, i.e., [path_to_DatasetA, path_to_DatasetB, ...]
    mask_roots: same as above, but the .png are masks predicted by the model
    strict: when True, all videos in the dataset must have corresponding predictions.
            Setting it to False is useful in cases where the ground-truth contains both train/val
                sets, but the model only predicts the val subset.
            Either way, if a video is predicted (i.e., the corresponding folder exists),
                then it must at least contain all the masks in the ground truth annotations.
                Masks that are in the prediction but not in the ground-truth
                (i.e., sparse annotations) are ignored.
    skip_first_and_last: whether we should skip the first and the last frame in evaluation.
                            This is used by DAVIS 2017 in their semi-supervised evaluation.
                            It should be disabled for unsupervised evaluation.
    """

    assert len(gt_roots) == len(mask_roots)
    single_dataset = len(gt_roots) == 1
    if evaluate_object_id_list is None:
        evaluate_object_id_list = range(1, 256)

    if verbose:
        if skip_first_and_last:
            print(
                "We are *SKIPPING* the evaluation of the first and the last frame (standard for semi-supervised video object segmentation)."
            )
        else:
            print(
                "We are *NOT SKIPPING* the evaluation of the first and the last frame (*NOT STANDARD* for semi-supervised video object segmentation)."
            )

    pool = Pool(num_processes)
    start = time.time()
    to_wait = []
    for gt_root, mask_root in zip(gt_roots, mask_roots):
        # Validate folders
        validated = True
        gt_videos = os.listdir(gt_root)
        mask_videos = os.listdir(mask_root)

        # if the user passed the root directory instead of Annotations
        if len(gt_videos) != len(mask_videos):
            if "Annotations" in gt_videos:
                if ".png" not in os.listdir(path.join(gt_root, "Annotations"))[0]:
                    gt_root = path.join(gt_root, "Annotations")
                    gt_videos = os.listdir(gt_root)

        # remove non-folder items
        gt_videos = list(filter(lambda x: path.isdir(path.join(gt_root, x)), gt_videos))
        mask_videos = list(
            filter(lambda x: path.isdir(path.join(mask_root, x)), mask_videos)
        )

        if not strict:
            videos = sorted(list(set(gt_videos) & set(mask_videos)))
        else:
            gt_extras = set(gt_videos) - set(mask_videos)
            mask_extras = set(mask_videos) - set(gt_videos)

            if len(gt_extras) > 0:
                print(
                    f"Videos that are in {gt_root} but not in {mask_root}: {gt_extras}"
                )
                validated = False
            if len(mask_extras) > 0:
                print(
                    f"Videos that are in {mask_root} but not in {gt_root}: {mask_extras}"
                )
                validated = False
            if not validated:
                print("Validation failed. Exiting.")
                exit(1)

            videos = sorted(gt_videos)

        print(f"In dataset {gt_root}, we are evaluating on {len(videos)} videos: {videos}")
        input_data = []
        for video in videos:
            input_data.append((video, evaluate_object_id_list))
        results = tqdm.tqdm(
            pool.imap(
                VideoEvaluator(
                    gt_root, mask_root, skip_first_and_last=skip_first_and_last
                ),
                input_data
            ),
            total=len(videos),
        )

    pool.close()

    all_global_jf, all_global_j, all_global_f, all_global_dice = [], [], [], []
    all_object_metrics = []
    for i, mask_root in enumerate(mask_roots):
        if not single_dataset:
            results = to_wait[i].get()

        all_iou = []
        all_boundary_f = []
        all_dice = []
        object_metrics = {}
        for name, iou, boundary_f, dice in results:
            all_iou.extend(list(iou.values()))
            all_boundary_f.extend(list(boundary_f.values()))
            all_dice.extend(list(dice.values()))
            object_metrics[name] = (iou, boundary_f, dice)
        global_j = np.array(all_iou).mean()
        global_f = np.array(all_boundary_f).mean()
        global_jf = (global_j + global_f) / 2
        global_dice = np.array(all_dice).mean()

        time_taken = time.time() - start
        """
        Build string for reporting results
        """
        # find max length for padding
        ml = max(*[len(n) for n in object_metrics.keys()], len("Global score"))
        # build header
        out_string = f'{"sequence":<{ml}},{"obj":>3}, {"J&F":>4}, {"J":>4}, {"F":>4}, {"Dice":>8}\n'
        out_string += (f'{"Global score":<{ml}},{"":>3}, {global_jf:.2f}, {global_j:.2f}, {global_f:.2f}, {global_dice:.2f}\n')
        # append one line for each object
        for name, (iou, boundary_f, dice) in object_metrics.items():
            for object_idx in iou.keys():
                j, f, d = iou[object_idx], boundary_f[object_idx], dice[object_idx]
                jf = (j + f) / 2
                out_string += (
                    f"{name:<{ml}},{object_idx:03}, {jf:>4.2f}, {j:>4.2f}, {f:>4.2f}, {d:>4.2f}\n"
                )

        # print to console
        if verbose:
            print(out_string.replace(",", " "), end="")
            print("\nSummary:")
            print(
                f"Global score: J&F    J        F     Dice"
            )
            print(
                f"              {global_jf:.2f}  {global_j:.2f}  {global_f:.2f}  {global_dice:.2f}"
            )
            print(f"Time taken: {time_taken:.2f}s")

        # print to file
        result_path = path.join(mask_root, "results.csv")
        print(f"Saving the results to {result_path}")
        with open(result_path, "a+") as f:
            f.write('Epoch: ' + str(epoch) + ' is below\n')
            f.write(out_string)
            f.write("\n")

        all_global_jf.append(global_jf)
        all_global_j.append(global_j)
        all_global_f.append(global_f)
        all_global_dice.append(global_dice)
        all_object_metrics.append(object_metrics)

    return all_global_jf, all_global_j, all_global_f, all_global_dice, all_object_metrics
