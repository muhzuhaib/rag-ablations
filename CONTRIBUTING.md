# Contributing

Issues, questions and pull requests are all welcome. This is a small project, so the process is
short.

## Where to put things

- **A question, or a result you want to discuss:**
  [Discussions](https://github.com/muhzuhaib/rag-ablations/discussions).
- **A number looks wrong, or something is broken:**
  [open an issue](https://github.com/muhzuhaib/rag-ablations/issues/new/choose).
- **A security problem:** do not open an issue. See [SECURITY.md](SECURITY.md).

## Running it

No API keys, no accounts and no GPU. Embeddings are computed locally.

```bash
git clone https://github.com/muhzuhaib/rag-ablations && cd rag-ablations
python -m pip install -e ".[dev,service]"
pytest -q
```

Reproducing a published figure is one command:

```bash
python -m rag_ablations.benchmark --dataset scifact --systems sparse --check
```

`--check` compares what it just measured against the committed results and fails if a figure has
moved. That gate runs on every push, and monthly on a schedule so a benchmark that quietly stopped
reproducing is noticed even in a month when nobody touched the repo.

## The one rule that matters here

**Every result is measured against a stated baseline, and the baseline is validated against
published figures.** Most of the value in this repo is that its BM25 baseline is checked against the
BEIR numbers instead of being asserted. A pull request that adds a retrieval method has to report it
the same way:

- Report **nDCG@10 and recall on the same corpus and the same queries** as everything already in the
  table. A number measured on a different slice is not comparable and cannot go in the table.
- **Commit the results file**, so the figure in the README has something behind it and `--check` can
  hold it there.
- **Keep the negative results.** Dense retrieval losing to BM25 on this corpus, and the reranker
  being neutral while roughly two orders of magnitude slower, are findings. Removing an unflattering
  row would make the flattering ones meaningless.
- **Report the cost as well as the score.** A method that wins by a rounding error and costs 200x the
  latency belongs in the table with that latency next to it.

## What a good pull request looks like

- One concern per pull request, with a test that fails without the change.
- If a figure in the README moves, say so in the description and in `CHANGELOG.md`, with the old
  value and the new one.
- Factual claims about a dataset or a published result cite the source they came from.

CI runs the tests on Python 3.10 and 3.13, re-runs the reproducibility gate, and builds the container
and queries it. All of it has to be green.

## Style

Nothing is enforced by a formatter. Match the surrounding code, and write comments that say why
something is the way it is rather than restating what the line does.
