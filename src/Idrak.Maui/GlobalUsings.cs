// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// MAUI's implicit usings bring Microsoft.Maui.Controls.Device (an obsolete MAUI class); the library's device is meant here.
global using Device = Idrak.Abstraction.Device;

// MAUI's Microsoft.Maui.Devices.DeviceInfo (the phone's model) is not the library's device listing entry.
global using DeviceInfo = Idrak.Diagnostics.DeviceInfo;
