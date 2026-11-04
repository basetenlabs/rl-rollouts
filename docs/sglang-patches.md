# Patching SGLang

> Custom models and SGLang patches need enablement for your organization; see [availability](../README.md#bring-your-own-model).

When your model needs SGLang changes (a model file, an attention backend, a Triton kernel, request fields,
scheduler code), put them in your package as `sglang.patch`: a plain `git diff` against the SGLang fork
that the image pins. The README's [current-image line](../README.md#bring-your-own-model) names the image and its SGLang release; download that
release from [rl-rollouts releases](https://github.com/basetenlabs/rl-rollouts/releases) and check it with `sha256sum -c sglang-<pin>.tar.gz.sha256`.
It contains the fork tree minus the protected files, plus `SGLANG_EXPORT.md`, which states the pin, lists
what is protected and shows how to make the patch:

```sh
cd sglang && git init -q && git add -A && git commit -qm "fork <pin>"
# edit python/sglang/...
git diff --no-renames -- python/sglang > ../<package>/sglang.patch
```

A model class goes in your package's `models/` directory, not in the patch: modules there register with
SGLang by its `EntryClass` convention, each only for an architecture your specs describe.
[`examples/joey-1`](../examples/joey-1/README.md) has both.

## Rules

Every process of the deployment's worker containers (the SGLang engine's, on every node) applies the patch
to the image's SGLang before anything imports it, and then serves the patched and new modules; the
frontend, router and planner containers neither fetch the package nor patch. Everything else stays the image's. The deployment stops before
any weights load if any of these is true:

- a hunk does not apply exactly where its header says;
- a file is not a `.py` file under `python/sglang/`, or the patch deletes, renames or changes the mode of a file;
- the patch touches a protected file (the weight-update library and update transaction), or changes, adds
  or removes a protected definition in a shared file (pause, update and preparation handlers and their
  request types, the runners an update writes, model-class binding, and what our runtime wraps); the export lists both;
- a patched file uses `__file__` (patched modules run from a private copy).

These checks catch accidental edits to the weight-update paths. They are not a security boundary: your
package's own code runs in the engine process. The startup verdict, the update checksum and the
conformance kit are what prove an update correct.

The patch is part of your package, so the kit report's package hash covers it, and the kit
runs with it applied. A patch is tied to one image: a new image with a new SGLang pin has a new release; rebase the
patch onto it and run the kit again. Compiled kernels are out of scope; Triton kernels in Python files can be patched.
