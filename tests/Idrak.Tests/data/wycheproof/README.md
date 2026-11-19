# Project Wycheproof test vectors

Test vectors from Project Wycheproof, <https://github.com/C2SP/wycheproof>, directory `testvectors_v1/`, at commit
`65ca494727a52944c204441c860b039e6786b99b` (2026-11-11). Each file is the original JSON, unchanged, compressed with
`gzip -9 -n`.

> Copyright 2016-2026 The Wycheproof Authors, offered under the Apache License 2.0
> (<https://www.apache.org/licenses/LICENSE-2.0>).

The test group `documents pdf: signatures: known bad` (tests/Idrak.Tests.Library/DocumentPdfSignatureVectorsTests.cs)
reads them: every "valid" case must verify, every "invalid" one must fail, and each "acceptable" one is listed with how
Idrak answered. The cases go through `CmsKeys.Verify` (and `CmsKeys.Recover`, the legacy PKCS #1 form's route); groups
with a digest Idrak does not read (SHA-224, SHA-512/224, SHA-512/256, SHA-3, SHAKE) are counted and named as not read.

| Files | What they check |
|---|---|
| `rsa_signature_{2048,3072,4096}_{sha256,sha384,sha512}_test.json.gz` | RSA PKCS #1 v1.5 verification: padding, DigestInfo encodings (BER, missing NULL), wrong hashes, malleability |
| `rsa_pkcs1_{2048,3072,4096}_sig_gen_test.json.gz` | RSA PKCS #1 v1.5 signatures that verify, with SHA-1 (2048) and SHA-2 |
| `rsa_pss_*_test.json.gz` | RSA-PSS verification: salt lengths 0 to 64, MGF1 digests, modified signatures |
| `rsa_pss_2048_sha256_mgf1_32_params_test.json.gz`, `rsa_pss_misc_params_test.json.gz` | RSA-PSS keys certified with their parameters (RFC 4055): the scheme read from the key |
| `ecdsa_{secp256r1,secp384r1,secp521r1,secp256k1,brainpoolP256r1,brainpoolP384r1,brainpoolP512r1}_*_test.json.gz` | ECDSA verification: the DER signature (BER, missing zeros, wrong types, trailing bytes), r and s out of range, edge-case keys and digests |
