// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Models;

namespace Idrak.Generation;

/// <summary>
/// How a chat generator reads a vision-language model's images (<see cref="Images"/>, for
/// <see cref="ChatGenerator.Images"/>), all from the model's vision family (<see cref="PretrainedModel.Vision"/>): the
/// encoder (built when the first image is read, disposed with this), the prompt format and the attention rule. Every
/// image is decoded by the library's codecs and turned upright by its EXIF orientation (<see cref="ChatImageDecoder"/>,
/// as transformers' load_image does), then goes through <see cref="ChatImages.Transforms"/> unless a request gives its own
/// (<see cref="ChatRequest.ImageTransforms"/>).
/// </summary>
public sealed class ModelImages : IDisposable
{
    private readonly LazyEncoder _encoder;

    private ModelImages(PretrainedModel model, PretrainedVision vision, VisionEncoderOptions options, ImageTransformPipeline? transforms,
        Func<PretrainedModel, VisionEncoderOptions, IVisionEncoder>? encoderFactory)
    {
        Options = options;
        _encoder = new LazyEncoder(() => (encoderFactory ?? ((m, o) => m.CreateVisionEncoder(o)))(model, options), vision.Width, model.Device);
        Images = new ChatImages(_encoder, vision.PromptFormat, vision.Attention)
        {
            Decode = ChatImageDecoder.Decode, Transforms = transforms ?? ImageTransformPipeline.Empty, Owner = this,     // an engine disposes it with the model it serves
        };
    }

    /// <summary>
    /// The image reading of <paramref name="model"/>, or null for a text model. <paramref name="transforms"/> run on every
    /// decoded image first (unless a request gives its own); <paramref name="visionOptions"/> are the family's options for
    /// every image (their keys checked against the family's). <paramref name="encoderFactory"/> makes the encoder instead
    /// of the model's own (<see cref="PretrainedModel.CreateVisionEncoder"/>: tests feed reference features with it).
    /// </summary>
    /// <exception cref="ArgumentException">A vision option the family does not know.</exception>
    public static ModelImages? For(PretrainedModel model, ImageTransformPipeline? transforms = null, VisionOptions? visionOptions = null,
        Func<PretrainedModel, VisionEncoderOptions, IVisionEncoder>? encoderFactory = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.Vision is not { } vision)
        {
            return null;
        }

        visionOptions?.ThrowIfUnknown(vision.Family, vision.VisionOptionKeys);
        return new ModelImages(model, vision, new VisionEncoderOptions { Device = model.Device, VisionOptions = visionOptions }, transforms, encoderFactory);
    }

    /// <summary>The device and vision options its encoder is built with.</summary>
    public VisionEncoderOptions Options { get; }

    /// <summary>What the chat generator reads images with.</summary>
    public ChatImages Images { get; }

    /// <summary>The family's encoder once an image has been read (it is built on first use), else null.</summary>
    public IVisionEncoder? Encoder => _encoder.Current;

    /// <summary>Disposes the encoder, when it was built.</summary>
    public void Dispose() => _encoder.Dispose();

    // The family's encoder, built when an image is first read (its layout or its features), disposed with the model.
    private sealed class LazyEncoder(Func<IVisionEncoder> create, int width, Device device) : IVisionEncoder
    {
        private readonly Lock _lock = new();
        private IVisionEncoder? _built;

        public int Width => width;

        public Device Device => device;

        public IVisionEncoder? Current
        {
            get
            {
                lock (_lock)
                {
                    return _built;
                }
            }
        }

        public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null) => Built().Blocks(image, options);

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images, VisionOptions? options = null) => Built().Encode(images, options);

        public void Dispose()
        {
            lock (_lock)
            {
                _built?.Dispose();
                _built = null;
            }
        }

        private IVisionEncoder Built()
        {
            lock (_lock)
            {
                return _built ??= create();
            }
        }
    }
}
