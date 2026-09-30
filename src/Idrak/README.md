# Idrak

Self-contained neural network library for .NET: tensors, autograd, layers and optimizers with a SIMD CPU backend and a CUDA backend that talks to the NVIDIA driver directly. No TensorFlow, no native dependencies.

## Install

```bash
dotnet add package Idrak
```

Runs on any machine with .NET 10 (SIMD CPU kernels) and on NVIDIA GPUs through the display driver alone: no CUDA Toolkit, cuBLAS or cuDNN. `Device.Default` picks the first GPU and falls back to the CPU.

## Example: predict house prices, written two ways

The library has two ways to write the same code. The **original API** spells out every step; the **simplified API**
makes the same calls with less code (a network builder, data extensions, one training record, a predictor). Both
train the same kind of network with the same settings on the same data.

### Original API

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

### Simplified API

```csharp
using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

var data = Dataset.LoadCsv("houses.csv", new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
var split = data.Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();   // scalers fitted on the training part

using var model = Network.Input(data.FeatureCount)
    .Linear(64).ReLU().Dropout(0.05f)
    .Linear(32).ReLU()
    .Linear(1)
    .Build();

TrainingHistory history = new TrainingRun
{
    Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 2e-3f),
    Train = split.Train.Batches(64, shuffle: true), Validation = split.Test.Batches(512), Epochs = 400,
    Metrics = [Metric.MeanAbsoluteError], EarlyStoppingPatience = 30,   // restores the best weights when it stops
}.Fit();

using var predictor = Predictor.For(model)
    .ScaleInputs(split.FeatureScaler!).UnscaleOutputs(split.TargetScaler!)   // raw features in, dollars out
    .Output(v => v[0])
    .Build();
float price = predictor.Predict([2291, 5, 3, 4, 35.7f, 8, 1, 0, 11952]);
```

`houses.csv` is a table of listings with numeric columns and a `price` column (the repository's HousePrices sample
has one with 2,500 rows).

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: MIT.
