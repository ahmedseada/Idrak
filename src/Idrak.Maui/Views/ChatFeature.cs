// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "chat" feature: a model from the model store chats, its answer streamed with its token count and tokens
/// per second, the first answer after a load (cold) reported apart from the warm ones. The families it reads are the app's
/// to register (each text plug-in by its own Register call); a model whose family is not registered is refused, named. Its
/// view model is the app's one (<c>UseIdrak</c> registers it once), so a conversation goes on while another page is open.
/// </summary>
public sealed class ChatFeature : IMobileFeature
{
    // Its name in MobileFeatures.
    private const string FeatureName = "chat";

    /// <inheritdoc />
    public string Name => FeatureName;

    /// <inheritdoc />
    public string Summary => "Chat with a model from the store: the answer streams with its tokens per second, cold and warm apart";

    /// <inheritdoc />
    public string Title => "Chat";

    /// <inheritdoc />
    public string Glyph => "❝";

    /// <inheritdoc />
    public int Order => 20;

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.Model;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = ["Idrak", "Idrak.Text"];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        if (services.GetService<ChatViewModel>() is { } shared)
        {
            return shared;
        }

        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        var models = services.GetService<IModelService>() ?? new ModelService(new ManualMemoryPressure(), options);
        return new ChatViewModel(services.GetService<IModelStore>() ?? new ModelStore(options), services.GetService<IChatModelService>() ?? new ChatModelService(models), options);
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new ChatPage(viewModel as ChatViewModel ?? throw new ArgumentException($"The chat page takes a {nameof(ChatViewModel)}.", nameof(viewModel)), services);
}
