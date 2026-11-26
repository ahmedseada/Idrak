// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;

// Plan 37, K0: a paged key/value cache with shared prompt prefixes. A context paged in a KeyValuePages pool gives the
// unpaged cache's logits bit for bit when fed alike (every built-in format); a context that reads another's published
// pages gives what an unpaged context holding the same rows gives; a shared page is copied before it is written; N
// contexts sharing a prompt hold about one prompt's pages and their own; the generator shares its generations' pages.
internal abstract partial class Tests
{
}
