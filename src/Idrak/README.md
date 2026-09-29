# Idrak

Self-contained neural network library for .NET: tensors, autograd, layers and optimizers with a SIMD CPU backend and a CUDA backend that talks to the NVIDIA driver directly. No TensorFlow, no native dependencies.

## Install

```bash
dotnet add package Idrak
```

Runs on any machine with .NET 10 (SIMD CPU kernels) and on NVIDIA GPUs through the display driver alone: no CUDA Toolkit, cuBLAS or cuDNN. `Device.Default` picks the first GPU and falls back to the CPU.

## Example

```csharp
using Idrak;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

var data = Dataset.LoadCsv("houses.csv", new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
var (train, test) = data.Split(0.8, seed: 1);

var x = StandardScaler.FitFeatures(train);
var y = StandardScaler.FitTargets(train);
var trainLoader = new DataLoader(train.Scale(x, y), batchSize: 64, shuffle: true);
var testLoader  = new DataLoader(test.Scale(x, y), batchSize: 512);

using var model = new Sequential
{
    new Linear(data.FeatureCount, 64), new ReLU(), new Dropout(0.05f),
    new Linear(64, 32), new ReLU(),
    new Linear(32, 1),
};

var trainer = new Trainer(model, new Adam(model.Parameters(), 2e-3f), Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
    EarlyStoppingPatience = 30,             // restores the best weights when it stops
};
TrainingHistory history = trainer.Fit(trainLoader, epochs: 400, validation: testLoader);
float[,] predictions = trainer.Predict(test.Scale(x, y));
```

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: MIT.
