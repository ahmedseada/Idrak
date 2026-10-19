# Tokenizer training references (plan 24, phase T4)

Written by `tools/tokenizers/make_training_fixtures.py` (with the versions pinned in `tools/tokenizers/requirements.txt`;
`manifest.json` records what ran), from `../tokenizers/corpus.txt` and `../pretrain/corpus.txt`, each line read as the
C# trainers read it (without its `\n`, a `\r` before it dropped: `LineEnds` "lf"). This file is written by hand; the
script empties the folder but for it.

| File | What | Held to |
|---|---|---|
| `unigram.json` | Per configuration (`unigram`: vocabulary 1000; `unigram-bytes`: byte fallback and split digits; `unigram-arabic`: vocabulary 800, tatweel and the diacritics removed by a sentencepiece rule, longest piece 8, shrinking 0.8, 3 EM steps), the options and every piece (text, score, type) sentencepiece 0.2.2's unigram trainer learned: one thread, no shuffling, the identity normalization, character coverage 0.9995, `<unk>` `<s>` `</s>` as ids 0-2. | `tokenizer training: unigram learns SentencePiece's vocabulary`: the special and byte pieces at the same ids, and the learned pieces overlapping above the floor in `TokenizerTrainingTests.cs` |
| `wordpiece.json` | The vocabulary (in id order) of tokenizers' WordPieceTrainer algorithm, ported to Python in the script with its ties fixed (`fixed`), and the one tokenizers 0.23.2 itself learned in the run that wrote the file (`huggingface`); vocabulary 2000, minimum frequency 2, BERT's five special tokens, lowercased. | `fixed` exactly, `huggingface` by overlap above the floor |
| `manifest.json` | Library versions, the inputs' and outputs' sizes and SHA-256. | |

tokenizers numbers the `##` characters in hash order, which changes from process to process, so its WordPiece vocabulary
changes from run to run (four runs on this text agreed with each other on 96.5-97.9% of their tokens, and with `fixed`
on 96.5-98.1%). Everything but `huggingface` is the same on every rerun with the same versions.
