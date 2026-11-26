// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Layers;
using Idrak.Models;

// Plan 11, "Vision contracts (families are applications)": image token formats, and the image tokens placed in a prompt.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionContractGroup =
    [
        ("vision contracts: image token formats expand by each image's blocks (several with Join's text); images whose tokens touch are told apart by their counts; M-RoPE position ids are carried to the prefill", VisionContractRules),
    ];

    private static void VisionContractRules(Device device)
    {
        // An image token format expands each marker by its own image's count, with or without begin and end tokens.
        var tokenizer = BpeTokenizer.Load(LlavaData("tiny-llava"));
        int image = tokenizer.Encode("<image>").Last();
        var llava = new ImageTokenFormat("llava", image, image);
        string two = llava.Expand("A <image> and <image>.", [[new ImageTokenLayout(2)], [new ImageTokenLayout(3)]], tokenizer);
        Check(two == "A <image><image> and <image><image><image>.", $"expanded by layouts: {two}");
        Check(Fails(() => llava.Expand("<image>", [[new ImageTokenLayout(1)], [new ImageTokenLayout(1)]], tokenizer)), "one marker for two images is refused");

        // An image of several blocks: refused by a format without Join, written by Join's text otherwise (each block whole).
        try
        {
            llava.Expand("A <image>.", [[new ImageTokenLayout(1), new ImageTokenLayout(2)]], tokenizer);
            Check(false, "several blocks without Join are refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("2 blocks", StringComparison.Ordinal) && e.Message.Contains("Join", StringComparison.Ordinal), e.Message);
        }

        var views = new ViewsFormat(image);
        string several = views.Expand("A <image>, <image>.", [[new ImageTokenLayout(1), new ImageTokenLayout(2), new ImageTokenLayout(1)], [new ImageTokenLayout(2)]], tokenizer);
        Check(several == "A whole [<image>] parts [<image><image>] [<image>], [<image><image>].", $"several blocks joined: {several}");
        Check(views.Block(new ImageTokenLayout(3), tokenizer) == "[<image><image><image>]", "one block's text");

        // Two images whose tokens touch (LLaVA writes <image><image>): told apart by their counts.
        using var a = Tensor.From(new float[2 * 4], [2, 4], device);
        using var b = Tensor.From(new float[3 * 4], [3, 4], device);
        int[] ids = [1, image, image, image, image, image, 9];
        var found = ImagePrefill.Locate(ids, image, [a, b]);
        Check(found is [{ Position: 1, Tokens: 2 }, { Position: 3, Tokens: 3 }], "adjacent images by their counts");
        Check(Fails(() => ImagePrefill.Locate(ids, image, [a, a])), "image tokens left over are refused");
        Check(Fails(() => ImagePrefill.Locate(ids, image, [b, b])), "too few image tokens are refused");
        Check(Fails(() => _ = new ImageTokenLayout(6) { Grid = [2, 2] }), "a grid that does not hold the tokens is refused");

        // The M-RoPE slot: position ids reach the prefill.
        using var features = new ImageFeatures(Tensor.From(new float[16 * 32], [16, 32], device), new ImageTokenLayout(16) { Grid = [4, 4] },
            Tensor.From(new float[3 * 16], [3, 16], device));
        int[] prompt = [.. Enumerable.Repeat(1, 3), .. Enumerable.Repeat(image, 16), 5];
        var placed = ImagePrefill.Locate(prompt, image, [features]);
        Check(placed[0].Positions is not null, "position ids are carried to the prefill");
    }

    // Views of an image joined as "whole … parts …".
    private sealed class ViewsFormat(int image) : ImageTokenFormat("views", image, image, before: "[", after: "]"), IImageBlocksJoin
    {
        public string Join(IReadOnlyList<string> blocks) => "whole " + blocks[0] + " parts " + string.Join(" ", blocks.Skip(1));
    }
}
