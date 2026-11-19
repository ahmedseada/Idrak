# x509-limbo path-validation cases

Test cases from Trail of Bits' x509-limbo, <https://github.com/C2SP/x509-limbo>, file `limbo.json` at commit
`b12423782a88abead01141fa6582254e3eb49e74` (2026-11-17), licensed under the Apache License 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>).

`limbo-paths.json.gz` holds the cases of the namespaces that test certification paths and revocation lists (`pathlen`,
`pathological`, `rfc5280`, `invalid`, `cve`, `crl`, `rfc9881`: 165 cases), each with the fields limbo.json gives it
except the leaf's private key and the expected peer names (`id`, `description`, `features`, `validation_kind`,
`trusted_certs`, `untrusted_intermediates`, `peer_certificate`, `validation_time`, `expected_result`, `max_chain_depth`,
`crls`), unchanged; the namespaces about web-server names (`bettertls`, `webpki`, `online`) are only counted. Made by
selecting from limbo.json, written as indented UTF-8 JSON and compressed with gzip (level 9, no time).

The test group `documents pdf: signatures: known bad` (tests/Idrak.Tests.Library/DocumentPdfSignatureVectorsTests.cs)
builds each applied case's path with Idrak's own path builder (the leaf as a document's signer, the intermediates as the
certificates a document carries, the trusted certificates as the app's trust, at the case's time) and reads each of its
lists, and requires the case's verdict. The cases a document's check does not apply, and why, are named in that test
and printed by name; none is skipped silently.
