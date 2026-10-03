# Items to review

Things left as they are for now that need a decision later.

## The chat API server in Idrak.AspNetCore

`src/Idrak.AspNetCore` (`MapChatApi` in IdrakEndpoints.cs, wire types in ChatApi.cs) serves Idrak's own models over
the local chat API common clients speak (`/api/chat`, `/api/tags`, `/api/ps`, `/api/version`), so those clients can
use an Idrak model. It is a server, not a client: the library does not call any other model server. It was kept when
the OpenAI-compatible client and the teacher paths were removed (commit 6d5efb3). Decided: kept, under
provider-neutral names (`MapChatApi`, `ChatApiOptions`, `ChatApi*` types; the former names are obsolete forwarders
for one release).

## Teacher-generated samples

Removed from the library entirely. Generating samples with an outside model (for example through Postman) is left to
the user, who brings the results in as a dataset. To revisit much later.
