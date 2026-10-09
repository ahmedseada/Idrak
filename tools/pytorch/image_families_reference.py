"""Writes tests/Idrak.Tests/data/image-families: two tiny image model checkpoints and PyTorch's results for them, so the
image model families registered by tests/Idrak.PluginTests (plan 13, step 6) can be checked against a reference.

Run: python tools/pytorch/image_families_reference.py   (torch 2.14.1 CPU, Pillow 12.3, safetensors 0.8, onnx 1.23.2).
Reruns write the same bytes. Checked by `IDRAK_FILTER="outside plug-in"` (the image family tests).

  images/*.png     three small images: colour 48x40, grey 30x30, colour 23x61 (shapes on gradients)
  tiny-resnet/     a ResNet-style classifier (torchvision's layout and tensor names: conv1, bn1, a max pool, one basic
                   block of 8 channels, one of 16 with a strided 1x1 downsample, global average pooling, fc to 5 classes)
                   as config.json (architecture "OutsideTinyResNetForImageClassification", id2label), model.safetensors
                   (convolution and fc weights stored as bfloat16, norms and biases as float32) and
                   preprocessor_config.json (shortest edge 36, bicubic, center crop 32x32, rescale, ImageNet mean/std)
  tiny-detector/   an anchor-free grid detector (three strided 3x3 convolutions with ReLU, weights scaled by 3, and a 1x1
                   head, weights by 2 with box-size and objectness biases set: per cell of an 8x8 grid of stride 8, tx,
                   ty, tw, th, objectness and 3 class logits) exported to model.onnx
                   (torch.onnx.export, dynamo=False, opset 17, the batch dynamic, metadata "architecture"), with
                   config.json (architecture "OutsideGridDetector", stride, id2label) and preprocessor_config.json
                   (resize to 64x64, bilinear, rescale, no normalization)

reference.json holds, per image: the classifier's pixel values (transformers' PIL path: Pillow's resize by the shortest
edge, center crop, the bytes times 1/255 in double precision then float32, (x - mean) / std in float32) and logits; the
detector's raw outputs and its decoded detections in image pixels, as rows [x, y, width, height, class, score]: per cell
cx = (gx + sigmoid(tx)) * stride, cy = (gy + sigmoid(ty)) * stride, w = exp(tw) * stride, h = exp(th) * stride, the best
class c and score sigmoid(objectness) * sigmoid(class c); boxes scaled to the image (W / 64, H / 64), clipped, empty ones
dropped, scores below 0.3 dropped, torchvision's batched_nms at IoU 0.5 (per class), best first. The detector's weights
come from the first seed whose candidates are all at least 0.001 from the score threshold and whose same-class pairs are
at least 0.001 from the IoU threshold, so float32 rounding cannot change which detections are kept.
"""

import json
import os

import numpy as np
import onnx
import torch
import torchvision
from PIL import Image
from safetensors.torch import save_file
from torch import nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "Idrak.Tests", "data", "image-families")
CLASSIFIER = "OutsideTinyResNetForImageClassification"
DETECTOR = "OutsideGridDetector"
MIN_SCORE, IOU = 0.3, 0.5


def values(t):
    # The shortest text that reads back as the same float32.
    return [float(str(np.float32(v))) for v in np.asarray(t, dtype=np.float32).reshape(-1).tolist()]


def write_json(path, value):
    with open(path, "w", newline="\n") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------- images

def make_images():
    os.makedirs(os.path.join(ROOT, "images"), exist_ok=True)
    rng = np.random.RandomState(7)

    def shapes(h, w, channels):
        y, x = np.mgrid[0:h, 0:w]
        base = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) * 128 // max(h + w - 2, 1))], axis=-1).astype(np.int32)
        for _ in range(4):
            cy, cx, r = rng.randint(0, h), rng.randint(0, w), rng.randint(3, max(4, min(h, w) // 3))
            colour = rng.randint(0, 256, size=3)
            if rng.randint(2):
                base[(np.abs(y - cy) <= r) & (np.abs(x - cx) <= r)] = colour
            else:
                base[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] = colour
        pixels = np.clip(base, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels[..., 0] if channels == 1 else pixels, "L" if channels == 1 else "RGB")

    images = {"a.png": shapes(40, 48, 3), "b.png": shapes(30, 30, 1), "c.png": shapes(61, 23, 3)}
    for name, image in images.items():
        image.save(os.path.join(ROOT, "images", name), optimize=False, compress_level=9)
    return {name: Image.open(os.path.join(ROOT, "images", name)) for name in images}


# transformers' PIL image processor steps (CLIPImageProcessor and the like), written out.
def preprocess(image, size=None, shortest_edge=None, crop=None, resample=Image.BILINEAR, mean=None, std=None):
    image = image.convert("RGB")
    w, h = image.size
    if shortest_edge is not None:
        short, long = min(w, h), max(w, h)
        new_long = int(shortest_edge * long / short)
        (nh, nw) = (new_long, shortest_edge) if w <= h else (shortest_edge, new_long)
    else:
        nh, nw = size
    image = image.resize((nw, nh), resample=resample)
    pixels = np.asarray(image)                                                  # [h, w, 3] bytes
    if crop is not None:
        ch, cw = crop
        top, left = (nh - ch) // 2, (nw - cw) // 2
        pixels = pixels[top:top + ch, left:left + cw]
    x = (pixels.astype(np.float64) * (1 / 255)).astype(np.float32)
    if mean is not None:
        x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


# ---------------------------------------------------------------- the classifier

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)) if stride != 1 or cin != cout else None

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + (x if self.downsample is None else self.downsample(x)))


class TinyResNet(nn.Module):
    def __init__(self, stem, widths, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, stem, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin = stem
        for i, width in enumerate(widths):
            setattr(self, f"layer{i + 1}", nn.Sequential(BasicBlock(cin, width, 1 if i == 0 else 2)))
            cin = width
        self.fc = nn.Linear(cin, classes)
        self.widths = widths

    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        for i in range(len(self.widths)):
            x = getattr(self, f"layer{i + 1}")(x)
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))


def classifier(images):
    folder = os.path.join(ROOT, "tiny-resnet")
    os.makedirs(folder, exist_ok=True)
    torch.manual_seed(0)
    labels = ["circle", "square", "triangle", "star", "ring"]
    model = TinyResNet(8, [8, 16], len(labels)).eval()
    stored = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
                module.running_mean.uniform_(-0.3, 0.3)
                module.running_var.uniform_(0.5, 2.0)
        for name, tensor in model.state_dict().items():
            if name.endswith("num_batches_tracked"):
                continue
            if tensor.dim() >= 2:                                               # convolution and fc weights: bfloat16, as stored
                tensor.copy_(tensor.to(torch.bfloat16).to(torch.float32))
                stored[name] = tensor.to(torch.bfloat16).contiguous()
            else:
                stored[name] = tensor.clone().contiguous()
    save_file(stored, os.path.join(folder, "model.safetensors"))
    write_json(os.path.join(folder, "config.json"), {
        "architectures": [CLASSIFIER], "num_channels": 3, "embedding_size": 8, "hidden_sizes": [8, 16], "depths": [1, 1],
        "layer_norm_eps": 1e-5, "id2label": {str(i): l for i, l in enumerate(labels)}, "label2id": {l: i for i, l in enumerate(labels)},
    })
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    write_json(os.path.join(folder, "preprocessor_config.json"), {
        "do_resize": True, "size": {"shortest_edge": 36}, "resample": 3, "do_center_crop": True, "crop_size": {"height": 32, "width": 32},
        "do_rescale": True, "rescale_factor": 1 / 255, "do_normalize": True, "image_mean": mean, "image_std": std, "do_convert_rgb": True,
    })
    result = {}
    with torch.no_grad():
        for name, image in images.items():
            x = preprocess(image, shortest_edge=36, crop=(32, 32), resample=Image.BICUBIC, mean=mean, std=std)
            result[name] = {"pixel_values": values(x), "logits": values(model(x))}
    return result


# ---------------------------------------------------------------- the detector

class GridDetector(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(), nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 16, 3, 2, 1), nn.ReLU())
        self.head = nn.Conv2d(16, 5 + classes, 1)

    def forward(self, x):
        return self.head(self.body(x))


def decode(out, stride, image_size, input_size):
    # out [5 + C, gh, gw] → candidate detections in image pixels (xyxy), their classes and scores.
    c5, gh, gw = out.shape
    gy, gx = torch.meshgrid(torch.arange(gh, dtype=torch.float32), torch.arange(gw, dtype=torch.float32), indexing="ij")
    cx = (gx + torch.sigmoid(out[0])) * stride
    cy = (gy + torch.sigmoid(out[1])) * stride
    w = torch.exp(out[2]) * stride
    h = torch.exp(out[3]) * stride
    probs = torch.sigmoid(out[4])[None] * torch.sigmoid(out[5:])
    score, cls = probs.max(0)
    W, H = image_size
    sx, sy = W / input_size[1], H / input_size[0]
    x1, y1 = (cx - w / 2) * sx, (cy - h / 2) * sy
    x2, y2 = x1 + w * sx, y1 + h * sy
    boxes = torch.stack([x1.clamp(0, W), y1.clamp(0, H), x2.clamp(0, W), y2.clamp(0, H)], -1).reshape(-1, 4)
    return boxes, cls.reshape(-1), score.reshape(-1)


def detector_outputs(model, images, input_size, stride):
    result, margins = {}, []
    with torch.no_grad():
        for name, image in images.items():
            x = preprocess(image, size=input_size, resample=Image.BILINEAR)
            out = model(x)[0]
            boxes, cls, score = decode(out, stride, image.size, input_size)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            keep = (area > 0) & (score >= MIN_SCORE)
            margins.append(float((score[area > 0] - MIN_SCORE).abs().min()))
            b, c, s = boxes[keep], cls[keep], score[keep]
            for k in c.unique():
                same = b[c == k]
                if len(same) > 1:
                    iou = torchvision.ops.box_iou(same, same)
                    off = iou[~torch.eye(len(same), dtype=torch.bool)]
                    margins.append(float((off - IOU).abs().min()))
            order = torchvision.ops.batched_nms(b, s, c, IOU)
            rows = [[float(b[i, 0]), float(b[i, 1]), float(b[i, 2] - b[i, 0]), float(b[i, 3] - b[i, 1]), float(c[i]), float(s[i])] for i in order]
            result[name] = {"outputs": values(out), "detections": values(np.asarray(rows, dtype=np.float32).reshape(-1)), "count": len(rows)}
    return result, min(margins)


def detector(images):
    folder = os.path.join(ROOT, "tiny-detector")
    os.makedirs(folder, exist_ok=True)
    labels = ["square", "disc", "bar"]
    input_size, stride = (64, 64), 8
    for seed in range(100):
        torch.manual_seed(seed)
        model = GridDetector(len(labels)).eval()
        with torch.no_grad():
            for layer in model.body:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.mul_(3.0)                                      # features that vary across the image
            model.head.weight.mul_(2.0)
            model.head.bias[2:4].fill_(0.8)                                     # boxes about 2.2 cells wide: neighbours overlap
            model.head.bias[4].fill_(-0.5)                                      # scores from about 0.1 to 0.75
        result, margin = detector_outputs(model, images, input_size, stride)
        if margin >= 0.001 and all(r["count"] > 0 for r in result.values()):
            break
    else:
        raise RuntimeError("no seed keeps every candidate away from the thresholds")

    path = os.path.join(folder, "model.onnx")
    torch.onnx.export(model, (torch.zeros(1, 3, *input_size),), path, dynamo=False, opset_version=17, input_names=["pixel_values"],
                      output_names=["outputs"], dynamic_axes={"pixel_values": {0: "batch"}, "outputs": {0: "batch"}})
    exported = onnx.load(path)
    entry = exported.metadata_props.add()
    entry.key, entry.value = "architecture", DETECTOR
    exported.producer_version = ""                                              # the same bytes whichever build exported it
    with open(path, "wb") as f:
        f.write(exported.SerializeToString())
    write_json(os.path.join(folder, "config.json"), {
        "architectures": [DETECTOR], "num_classes": len(labels), "stride": stride, "image_size": list(input_size),
        "id2label": {str(i): l for i, l in enumerate(labels)}, "label2id": {l: i for i, l in enumerate(labels)},
    })
    write_json(os.path.join(folder, "preprocessor_config.json"), {
        "do_resize": True, "size": {"height": input_size[0], "width": input_size[1]}, "resample": 2,
        "do_rescale": True, "rescale_factor": 1 / 255, "do_normalize": False, "do_convert_rgb": True,
    })
    return result, seed, margin


def main():
    torch.use_deterministic_algorithms(True)
    images = make_images()
    classes = classifier(images)
    detections, seed, margin = detector(images)
    reference = {"torch": torch.__version__.split("+")[0], "classifier": classes, "detector": detections, "detector_seed": seed,
                 "min_score": MIN_SCORE, "iou": IOU, "margin": round(margin, 4)}
    with open(os.path.join(ROOT, "reference.json"), "w", newline="\n") as f:
        f.write(json.dumps(reference, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"detector seed {seed}, margin {margin:.4f}; detections {[r['count'] for r in detections.values()]}")


if __name__ == "__main__":
    main()
