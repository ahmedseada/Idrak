// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Text.Generation;

// edit_file's tolerance (CodingToolOptions.EditTolerance), its error that quotes old_text and the file, the encoding kept,
// and the agent's reminder of failed changes (AgentOptions.FailedCallNudges): from Craft's agent bench, where a 7B
// Instruct model sent an old_text it had guessed before reading Program.cs (a byte order mark, CRLF, a blank last
// line) and was told only to read the file again, ten times; and answered "done" with its edits failed.
internal abstract partial class DesktopTests
{
}
