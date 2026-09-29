# Items to review

Things left as they are for now that need a decision later.

## Ollama-compatible server in Idrak.AspNetCore

`src/Idrak.AspNetCore` (`MapOllamaApi` in IdrakEndpoints.cs, wire types in Ollama.cs) serves
Idrak's own models over an Ollama-compatible HTTP API (`/api/chat`, `/api/tags`, `/api/ps`, `/api/version`), so
other tools that speak Ollama's protocol can use a Idrak model. It is a server, not a client: the library does
not call Ollama or any other model server. It was kept when the OpenAI-compatible client and the teacher paths were
removed (commit 6d5efb3). To decide: keep, change or remove.

## Teacher-generated samples

Removed from the library entirely. Generating samples with an outside model (for example through Postman) is left to
the user, who brings the results in as a dataset. To revisit much later.
