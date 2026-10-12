# Idrak.Imaging

Images for Idrak (no dependencies beyond Idrak): PNG, JPEG, BMP and Netpbm decoders and encoders written in C#, Pillow's
resampling, filters and enhancements byte for byte, the preprocessors of image models (resize, crop, rescale, normalize,
on the device too), chat image decoding, sample augmentations and image-folder datasets.

## Install

```bash
dotnet add package Idrak.Imaging
```

## What is in the package

| Area | What is there |
|------|---------------|
| Codecs | `PngCodec`, `JpegCodec` (baseline and progressive, decoded to Pillow's RGB), `BmpCodec`, `NetpbmCodec`; `PngEncoder`, `JpegEncoder`, `NetpbmEncoder`; registered in `ImageCodecs` (Idrak.Abstraction) when the registry is first used, so `ImageCodecs.Decode(path)` reads them |
| Transforms | `PillowImageOps` (resize with Pillow's filters, rank filters, blur, blends, point maps), the named transforms of `ImageTransforms` ("grayscale", "max_width", "contrast", ...) |
| Preprocessing | `ImagePreprocessor`: a model family's resize, crop, rescale and normalization from its preprocessor config, on the host or the device |
| Chat images | `ChatImageDecoder`: the images of a chat request (data URLs, files, bytes) decoded once |
| Training | `ImageFolderSource` (class folders, and the "images" source of `SampleSources`), augmentations for the `DataLoader` (`RandomFlip`, `RandomShift`, `RandomRotation`, `GaussianNoise`; "flip" and "shift" in `Augmentations`) |

## Contracts

The plug-in points only this package uses are in `Idrak.Imaging.Abstractions`: `ImageEncoders` and `IImageEncoder` (image
file encoders by format). The shared contracts (`ImageCodecs`, `ImageTransforms`, `Augmentations`, `SampleSources`,
`ImageData`) are in Idrak.Abstraction, and this package registers its built-ins into them.
