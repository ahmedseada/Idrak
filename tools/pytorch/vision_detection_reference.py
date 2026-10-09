"""Writes tests/Idrak.Tests/data/vision-detection/reference.json: the values torchvision, PyTorch, SciPy and pycocotools
give for what Idrak.Vision trains and measures detection and segmentation with, so Idrak can check its own against them.

Run: python tools/pytorch/vision_detection_reference.py
     (torch 2.14.1 CPU, torchvision 0.29.1, scipy 1.18.1, pycocotools 2.0.11, numpy). Reruns write the same bytes.
Checked by `IDRAK_FILTER="vision detection"`.

The inputs are drawn from numpy's default_rng(0) and written out, so the test does not reproduce the random numbers.

  box_losses     torchvision's generalized_box_iou_loss, distance_box_iou_loss and complete_box_iou_loss (and 1 - IoU from
                 its _loss_inter_union) with reduction none, and d(sum of losses)/d(predicted boxes), on random pairs and
                 three made ones (the same box, boxes apart, one inside the other)
  smooth_l1      F.smooth_l1_loss with beta 1 and 0.5 and F.l1_loss, reduction none, and the gradient of the sum (beta 1)
  focal          torchvision's sigmoid_focal_loss (alpha 0.25, gamma 2; alpha -1, gamma 0.5), reduction none, and the
                 gradient of the mean
  dice           soft dice per sample and class, 1 - (2 sum p t + 1) / (sum p + sum t + 1) on softmax probabilities, and
                 the gradient of the mean with respect to the logits
  hungarian      scipy.optimize.linear_sum_assignment on random costs (square and rectangular)
  matcher        torchvision's detection Matcher(0.5, 0.4) with and without allow_low_quality_matches on IoU matrices
  coco           pycocotools' COCOeval (bbox) stats on random scenes with crowds and empty images
  voc            Pascal VOC's voc_eval (every recall point, and VOC 2007's 11 points) in plain Python, in continuous
                 pixel coordinates (the devkit adds 1 to integer pixel extents; Idrak's VOC reader turns those into
                 continuous boxes)
  miou           the confusion-matrix mean IoU, pixel accuracy, mean class accuracy and frequency-weighted IoU with an
                 ignored class
  rle            pycocotools' mask.encode of binary masks (the compressed counts and their run lengths)
"""

import contextlib
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection._utils import Matcher
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss, sigmoid_focal_loss
from torchvision.ops._utils import _loss_inter_union

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "Idrak.Tests", "data", "vision-detection")


def values(t):
    a = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return [float(str(np.float32(v))) for v in a.reshape(-1).tolist()]


def doubles(a):
    return [float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1).tolist()]


def boxes(rng, n):
    xy = rng.uniform(0, 30, (n, 2))
    wh = rng.uniform(1, 20, (n, 2))
    return np.concatenate([xy, xy + wh], 1).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.use_deterministic_algorithms(True)
    reference = {"torch": torch.__version__.split("+")[0], "torchvision": torchvision.__version__.split("+")[0]}

    # ---- box losses: random pairs, plus the same box, boxes apart and one inside the other
    target = boxes(rng, 16)
    predicted = (target + rng.uniform(-4, 4, target.shape)).astype(np.float32)
    predicted[:, 2:] = np.maximum(predicted[:, 2:], predicted[:, :2] + 0.5)
    predicted[0] = target[0]
    predicted[1] = target[1] + 60
    predicted[2] = [target[2, 0] + 0.25, target[2, 1] + 0.5, target[2, 2] - 0.25, target[2, 3] - 0.5]
    case = {"predicted": values(predicted), "target": values(target)}
    for name, fn in [("giou", generalized_box_iou_loss), ("diou", distance_box_iou_loss), ("ciou", complete_box_iou_loss), ("iou", None)]:
        p = torch.tensor(predicted, requires_grad=True)
        t = torch.tensor(target)
        if fn is None:
            inter, union = _loss_inter_union(p, t)
            loss = 1 - inter / (union + 1e-7)
        else:
            loss = fn(p, t, reduction="none", eps=1e-7)
        loss.sum().backward()
        case[name] = {"losses": values(loss), "grad": values(p.grad)}
    reference["box_losses"] = case

    # ---- smooth L1 and L1
    a = torch.tensor(rng.normal(0, 2, (6, 4)).astype(np.float32), requires_grad=True)
    b = torch.tensor(rng.normal(0, 2, (6, 4)).astype(np.float32))
    smooth = F.smooth_l1_loss(a, b, beta=1.0, reduction="none")
    smooth.sum().backward()
    reference["smooth_l1"] = {"predicted": values(a), "target": values(b), "beta1": values(smooth), "grad_beta1": values(a.grad),
                              "beta05": values(F.smooth_l1_loss(a, b, beta=0.5, reduction="none")), "l1": values(F.l1_loss(a, b, reduction="none"))}

    # ---- focal loss
    logits = rng.normal(0, 3, (5, 7)).astype(np.float32)
    labels = (rng.uniform(0, 1, (5, 7)) < 0.3).astype(np.float32)
    focal = {"logits": values(logits), "targets": values(labels)}
    for key, alpha, gamma in [("a025_g2", 0.25, 2.0), ("none_g05", -1.0, 0.5)]:
        x = torch.tensor(logits, requires_grad=True)
        loss = sigmoid_focal_loss(x, torch.tensor(labels), alpha=alpha, gamma=gamma, reduction="none")
        loss.mean().backward()
        focal[key] = {"alpha": alpha, "gamma": gamma, "losses": values(loss), "grad_mean": values(x.grad)}
    reference["focal"] = focal

    # ---- dice on softmax probabilities, per sample and class
    seg_logits = rng.normal(0, 1.5, (2, 3, 4, 5)).astype(np.float32)
    seg_classes = rng.integers(0, 3, (2, 4, 5))
    onehot = np.transpose(np.eye(3, dtype=np.float32)[seg_classes], (0, 3, 1, 2))
    x = torch.tensor(seg_logits, requires_grad=True)
    p = x.softmax(1)
    t = torch.tensor(onehot)
    inter = (p * t).flatten(2).sum(2)
    total = p.flatten(2).sum(2) + t.flatten(2).sum(2)
    dice = 1 - (2 * inter + 1) / (total + 1)
    dice.mean().backward()
    reference["dice"] = {"logits": values(seg_logits), "targets": values(onehot), "losses": values(dice), "grad_logits_mean": values(x.grad)}

    # ---- Hungarian assignment
    hungarian = []
    for rows, cols in [(4, 4), (3, 6), (7, 2), (9, 9)]:
        cost = rng.uniform(-1, 1, (rows, cols))
        r, c = linear_sum_assignment(cost)
        hungarian.append({"rows": rows, "columns": cols, "cost": doubles(cost), "row_ind": r.tolist(), "col_ind": c.tolist(), "total": float(cost[r, c].sum())})
    reference["hungarian"] = hungarian

    # ---- threshold matching (torchvision's Matcher takes [truths, predictions])
    truths = boxes(rng, 5)
    anchors = np.concatenate([boxes(rng, 6), (truths + rng.uniform(-3, 3, truths.shape)).astype(np.float32)])
    anchors[:, 2:] = np.maximum(anchors[:, 2:], anchors[:, :2] + 0.5)
    iou = torchvision.ops.box_iou(torch.tensor(anchors), torch.tensor(truths))      # [predictions, truths]
    reference["matcher"] = {
        "predictions": len(anchors), "truths": len(truths), "quality": values(iou),
        "plain": Matcher(0.5, 0.4, allow_low_quality_matches=False)(iou.T.clone()).tolist(),
        "low_quality": Matcher(0.5, 0.4, allow_low_quality_matches=True)(iou.T.clone()).tolist(),
    }

    # ---- COCO mean average precision
    images, annotations, detections = [], [], []
    aid = 1
    for image_id in range(1, 9):
        images.append({"id": image_id, "file_name": f"{image_id}.png", "width": 640, "height": 480})
        if image_id == 8:
            continue                                                  # an image without objects (its detections are all wrong)
        for _ in range(int(rng.integers(1, 6))):
            w, h = float(rng.uniform(8, 200)), float(rng.uniform(8, 200))
            x, y = float(rng.uniform(0, 640 - w)), float(rng.uniform(0, 480 - h))
            crowd = int(rng.uniform() < 0.1)
            area = w * h * float(rng.uniform(0.6, 1.0))
            category = int(rng.integers(1, 4))
            annotations.append({"id": aid, "image_id": image_id, "category_id": category, "bbox": [x, y, w, h], "area": area, "iscrowd": crowd})
            aid += 1
            for _ in range(int(rng.integers(0, 3))):
                j = float(rng.uniform(0, 0.4))
                box = [x + rng.uniform(-0.5, 0.5) * j * w, y + rng.uniform(-0.5, 0.5) * j * h, w * (1 + rng.uniform(-0.5, 0.5) * j), h * (1 + rng.uniform(-0.5, 0.5) * j)]
                detections.append({"image_id": image_id, "category_id": category if rng.uniform() > 0.15 else int(rng.integers(1, 4)),
                                   "bbox": [float(v) for v in box], "score": float(rng.uniform())})
    for _ in range(12):
        w, h = float(rng.uniform(5, 100)), float(rng.uniform(5, 100))
        detections.append({"image_id": int(rng.integers(1, 9)), "category_id": int(rng.integers(1, 4)),
                           "bbox": [float(rng.uniform(0, 500)), float(rng.uniform(0, 380)), w, h], "score": float(rng.uniform())})
    # Round to float32, the precision Idrak's boxes have, so both sides see the same numbers.
    for a in annotations:
        a["bbox"] = [float(np.float32(v)) for v in a["bbox"]]
        a["area"] = float(np.float32(a["area"]))
    for d in detections:
        d["bbox"] = [float(np.float32(v)) for v in d["bbox"]]
        d["score"] = float(np.float32(d["score"]))
    gt_json = {"images": images, "annotations": annotations, "categories": [{"id": c, "name": f"c{c}"} for c in (1, 2, 3)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = json.loads(json.dumps(gt_json))
        gt.createIndex()
        dt = gt.loadRes(json.loads(json.dumps(detections)))
        evaluation = COCOeval(gt, dt, "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    reference["coco"] = {"ground_truth": gt_json, "detections": detections, "stats": doubles(evaluation.stats),
                         "stats_names": ["map", "map50", "map75", "map_small", "map_medium", "map_large", "mar1", "mar10", "mar100", "mar_small", "mar_medium", "mar_large"]}

    # ---- Pascal VOC average precision (voc_eval, continuous coordinates)
    voc_images = []
    for _ in range(6):
        objects = []
        for _ in range(int(rng.integers(1, 5))):
            x1, y1 = float(rng.uniform(0, 300)), float(rng.uniform(0, 300))
            objects.append({"box": [x1, y1, x1 + float(rng.uniform(10, 120)), y1 + float(rng.uniform(10, 120))], "class": int(rng.integers(0, 2)),
                            "difficult": bool(rng.uniform() < 0.15)})
        dets = []
        for o in objects:
            for _ in range(int(rng.integers(0, 3))):
                b = o["box"]
                w, h = b[2] - b[0], b[3] - b[1]
                j = float(rng.uniform(0, 0.5))
                x1, y1 = b[0] + float(rng.uniform(-0.5, 0.5)) * j * w, b[1] + float(rng.uniform(-0.5, 0.5)) * j * h
                dets.append({"box": [x1, y1, x1 + w * (1 + float(rng.uniform(-0.3, 0.3)) * j), y1 + h * (1 + float(rng.uniform(-0.3, 0.3)) * j)],
                             "class": o["class"], "score": float(rng.uniform())})
        dets.append({"box": [float(rng.uniform(0, 300)), float(rng.uniform(0, 300)), 0, 0], "class": int(rng.integers(0, 2)), "score": float(rng.uniform())})
        dets[-1]["box"][2], dets[-1]["box"][3] = dets[-1]["box"][0] + 30, dets[-1]["box"][1] + 30
        for o in objects:
            o["box"] = [float(np.float32(v)) for v in o["box"]]
        for d in dets:
            d["box"] = [float(np.float32(v)) for v in d["box"]]
            d["score"] = float(np.float32(d["score"]))
        voc_images.append({"objects": objects, "detections": dets})
    reference["voc"] = {"images": voc_images, "all_points": [voc_ap(voc_images, c, False) for c in (0, 1)], "eleven_points": [voc_ap(voc_images, c, True) for c in (0, 1)]}

    # ---- segmentation: confusion matrix with class 255 ignored
    classes = 4
    truth = rng.integers(0, classes, (3, 6, 7))
    truth[0, 0, :3] = 255
    pred = np.where(rng.uniform(0, 1, truth.shape) < 0.3, rng.integers(0, classes, truth.shape), truth)
    pred = np.where(truth == 255, rng.integers(0, classes, truth.shape), pred)
    keep = truth != 255
    confusion = np.bincount(classes * truth[keep] + pred[keep], minlength=classes * classes).reshape(classes, classes)
    inter = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - inter
    iou = inter / union
    accuracy = inter / confusion.sum(1)
    frequency = confusion.sum(1) / confusion.sum()
    reference["miou"] = {"classes": classes, "ignore": 255, "truth": truth.reshape(-1).tolist(), "predicted": pred.reshape(-1).tolist(), "height": 6, "width": 7,
                         "iou": doubles(iou), "miou": float(np.nanmean(iou)), "pixel_accuracy": float(inter.sum() / confusion.sum()),
                         "mean_accuracy": float(np.nanmean(accuracy)), "fwiou": float((frequency * iou).sum())}

    # ---- COCO run-length encoding
    rle_cases = []
    for h, w in [(5, 4), (9, 13)]:
        m = (rng.uniform(0, 1, (h, w)) < 0.4).astype(np.uint8)
        encoded = mask_utils.encode(np.asfortranarray(m))
        runs, inside, count = [], 0, 0
        for v in m.T.reshape(-1):
            if v != inside:
                runs.append(count)
                count, inside = 0, v
            count += 1
        runs.append(count)
        rle_cases.append({"height": h, "width": w, "mask": m.reshape(-1).tolist(), "counts": encoded["counts"].decode("ascii"), "runs": runs,
                          "area": int(mask_utils.area(encoded))})
    reference["rle"] = rle_cases

    with open(os.path.join(OUT, "reference.json"), "w", newline="\n") as f:
        f.write("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(reference[k], sort_keys=True, separators=(',', ':'))}" for k in sorted(reference)) + "\n}\n")
    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}: wrote {os.path.normpath(OUT)}")


def iou_continuous(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def voc_ap(images, cls, use_07):
    """voc_eval (py-faster-rcnn's port of the devkit) for one class, in continuous coordinates."""
    records = []
    npos = 0
    for k, image in enumerate(images):
        objs = [o for o in image["objects"] if o["class"] == cls]
        npos += sum(1 for o in objs if not o["difficult"])
        records.append({"objs": objs, "det": [False] * len(objs)})
    dets = [(k, d) for k, image in enumerate(images) for d in image["detections"] if d["class"] == cls]
    order = sorted(range(len(dets)), key=lambda i: -dets[i][1]["score"])
    tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
    for rank, i in enumerate(order):
        k, d = dets[i]
        record = records[k]
        overlaps = [iou_continuous(d["box"], o["box"]) for o in record["objs"]]
        ovmax = max(overlaps) if overlaps else -np.inf
        if ovmax > 0.5:
            j = int(np.argmax(overlaps))
            if not record["objs"][j]["difficult"]:
                if not record["det"][j]:
                    tp[rank] = 1
                    record["det"][j] = True
                else:
                    fp[rank] = 1
        else:
            fp[rank] = 1
    fp, tp = np.cumsum(fp), np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    if use_07:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
            ap += p / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


if __name__ == "__main__":
    main()
