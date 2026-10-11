# Optional vLLM patches

Off by default. Turn one on by listing its name in `recipe.yaml`:

```yaml
server:
  patches: hermes-chat          # space-separated, applied in this order
```

`qsa-logits-workspace` is built into the v5.2 image; list it only with the v5.1 image (`git checkout v5.1`).

At launch `run.sh` copies each file a patch touches out of the image, applies the patch and mounts the result
read-only over the image's file. The image is never changed; an empty
`patches` = the stock image. A patch that does not fit the image stops `run.sh` before anything starts.

| patch | what it does |
|---|---|
| `hermes-chat` | Hermes agent: reads its `{"reasoning": {…}}` object (thinking on/off, effort) and makes an omitted temperature greedy. Contributed by [@yume-arasaki](https://github.com/yume-arasaki) (#2) |
| `qsa-logits-workspace` | Long prefills: the QSA indexer reuses one logits workspace instead of growing a new buffer per chunk (host freezes on unified memory). Backport of [vllm-project/vllm#57105](https://github.com/vllm-project/vllm/pull/57105); reported by [@anzax](https://github.com/anzax) (#5) |
| `recoverssm-exact-commit` | RecoverSSM: the GDN state commit replays the accepted tokens forward with the verify step's own decay, so the committed state is bit-identical to per-token decoding (the closed form is not, on a GB10). Same cost. Backport of commit 514102a of [vllm-project/vllm#58863](https://github.com/vllm-project/vllm/pull/58863) (open upstream) |

## Adding one

A unified diff with paths relative to the `vllm` package (`--- a/entrypoints/…`, `+++ b/entrypoints/…`), made against
the image in `recipe.yaml`. Lines before the first `---` are a free-text description.
