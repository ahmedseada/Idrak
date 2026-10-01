// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// House-price Web API in a few lines: the inference engine from Idrak.AspNetCore serves the package saved by the
// HousePrices sample (network, weights and both scalers in one .ikm file).
//
//   dotnet run -c Release --project samples/Idrak.Samples.HousePrices                 # train; writes models/house-price.ikm
//   dotnet run -c Release --project samples/Idrak.Samples.HouseApi -- --Package <path to house-price.ikm>
//   curl localhost:5090/predict/house-price -H 'Content-Type: application/json' \
//        -d '{"area":2100,"bedrooms":4,"bathrooms":2,"age":15,"distanceKm":9.5,"quality":7,"garageSpaces":2,"hasPool":0,"lotSqft":6500}'
//   curl localhost:5090/predict/house-price/batch -H 'Content-Type: application/json' -d '[{...}, {...}]'
//   curl localhost:5090/status

using Idrak.AspNetCore;
using Idrak.Samples.HouseApi;

var builder = WebApplication.CreateBuilder(args);
string package = builder.Configuration["Package"]
    ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Idrak.Samples.HousePrices", "bin", "Release", "net10.0", "models", "house-price.ikm");

builder.Services.AddIdrak()
    .AddPredictor<House, PriceEstimate>("house-price", package, p => p
        .Input<House>(h => [h.Area, h.Bedrooms, h.Bathrooms, h.Age, h.DistanceKm, h.Quality, h.GarageSpaces, h.HasPool, h.LotSqft])
        .Output(v => new PriceEstimate(MathF.Round(v[0], 0)))
        .Batching(maxBatch: 256, maxWait: TimeSpan.FromMilliseconds(2)));

var app = builder.Build();
app.MapPredictor<House, PriceEstimate>("/predict/house-price", "house-price");
app.MapIdrakStatus("/status");
app.Run(builder.Configuration["Urls"] is null ? "http://localhost:5090" : null);

namespace Idrak.Samples.HouseApi
{
    /// <summary>A house to price: the nine features of the HousePrices dataset, in its column order.</summary>
    public sealed record House(float Area, float Bedrooms, float Bathrooms, float Age, float DistanceKm, float Quality,
        float GarageSpaces, float HasPool, float LotSqft);

    /// <summary>The model's answer.</summary>
    public sealed record PriceEstimate(float Price);
}
