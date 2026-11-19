# License scope and third-party assets

[日本語](THIRD_PARTY.md)

PascaLLM's original code and documentation are copyright 2026 cardy0828 and distributed under the [MIT License](LICENSE). Preserve applicable notices and terms for third-party components.

- **purego.** This project pins github.com/ebitengine/purego v0.11.2, licensed under Apache-2.0. When distributing a Go binary, include applicable dependency notices and license texts. A copy of the purego license is supplied. [Included license](licenses/purego-LICENSE) · [Upstream license](https://github.com/ebitengine/purego/blob/v0.11.2/LICENSE)
- **Models and evaluation assets.** Obtain weights, tokenizers, MTP assistants, and evaluation data separately. Check the provider, revision, usage restrictions, and redistribution requirements for each asset. Converting weights to PQ4 or PQ8 preserves applicable model terms. The engine's MIT license grants rights only to the covered engine materials.
- **Toolchain and generated assets.** Go runtime and standard library, NVIDIA driver and toolkit, llama.cpp, and Python tools each have their own terms. The repository includes sm_61 cubins generated from its CUDA sources, hashes, and compiler version. Retain applicable notices when distributing binaries or bundled tools. [Go license](https://go.dev/LICENSE)
