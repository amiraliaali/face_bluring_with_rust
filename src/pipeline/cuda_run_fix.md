# Rust + ONNX Runtime (ort) + CUDA: Full Debug/Resolution Log (End‑to‑End)

This document reconstructs **every major problem encountered**, the **symptom/error**, the **root cause**, and the **exact command(s) / code changes** that moved us to the next step — until ONNX Runtime was confirmed running on the GPU (CUDA) with `nvidia-smi`.

> Context:
>
> - Project: Rust video pipeline running a YOLOv8 face detector ONNX model.
> - Migration: from `onnxruntime` crate usage to the newer `ort = 2.0.0-rc.11` crate and enabling CUDA execution provider.

---

## 0) Starting point

### Cargo.toml (initial)

```toml
[package]
name = "pipeline"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1.0.101"
onnxruntime = "0.0.14"
ort = "=2.0.0-rc.11"
image = "0.24"
ndarray = "0.17"
opencv = "0.98.1"
```

### Initial code behavior goal

- Load `yolov8n-face.onnx`
- Read a video with OpenCV
- Run inference per frame
- Postprocess (NMS)
- Draw black rectangles over detected faces
- Write output video

---

## 1) Problem: `Session::run` API changed (generics + inputs type mismatch)

### Symptom / Errors

When compiling after switching toward `ort` session usage:

```
error[E0107]: method takes 1 generic argument but 3 generic arguments were supplied
  --> src/main.rs:83:31
   |
83 | let outputs = session.run::<f32, f32, _>(vec![input_tensor])?;
   |                           ^^^

error[E0277]: the trait bound `SessionInputs<...>: From<Vec<ArrayD<f32>>>` is not satisfied

error[E0599]: no method named `index_axis` found for reference `&ValueRef<'_>`
```

### Root cause

- In `ort = 2.0.0-rc.11`, `Session::run`:
  - no longer takes the old generic parameters like `run::<f32,f32,_>(...)`
  - expects **named inputs** (or a specific `SessionInputs` type), not `Vec<ArrayD<_>>`.
- Output values are not `ndarray` views directly; you must **extract** tensors from ORT values.

### Fix (code changes)

1. Build an ORT `Session` (new builder API)
2. Create input as `ort::value::Value` via `Value::from_array(...)`
3. Call `session.run(vec![(input_name, input_value)])`
4. Extract output via `try_extract_tensor::<f32>()`
5. Convert extracted `(shape, data)` to `ndarray::ArrayViewD`

---

## 2) Problem: Accessing session inputs (private fields)

### Symptom / Errors

```
error[E0616]: field `inputs` of struct `Session` is private
  --> src/main.rs:18:30
   |
18 | let input_name = session.inputs[0].name.clone();
   |                              ^^^^^ private field
help: a method `inputs` also exists, call it with parentheses
```

Then after changing to `inputs()`:

```
error[E0616]: field `name` of struct `Outlet` is private
  --> src/main.rs:23:42
   |
23 | let input_name = session.inputs()[0].name.clone();
   |                                          ^^^^ private field
help: a method `name` also exists, call it with parentheses
```

### Root cause

- In this `ort` version, fields are encapsulated:
  - `session.inputs` is private → you must use `session.inputs()`
  - `Outlet.name` is private → you must call `name()`

### Fix (code change)

```rust
let input_name = session.inputs()[0].name().to_string();
```

---

## 3) Problem: `Value::from_array` signature changed

### Symptom / Errors

```
error[E0061]: this function takes 1 argument but 2 arguments were supplied
  --> src/main.rs:81:27
81 | let input_value = Value::from_array(session.allocator(), &input_tensor)?;
   |                   ^^^^^^^^^^^^^^^^^                      ------------- unexpected argument

error[E0277]: the trait bound `&ort::memory::Allocator: OwnedTensorArrayData<_>` is not satisfied
```

### Root cause

- In `ort 2.0.0-rc.11`, `Value::from_array(...)` takes **only** the array-like input.
- You do **not** pass the allocator.

### Fix (code change)

```rust
let input_value = Value::from_array(input_tensor)?;
```

---

## 4) Problem: Output is not an ndarray view (no `.view()`)

### Symptom / Error

```
error[E0599]: no method named `view` found for tuple `(&ort::tensor::Shape, &[f32])`
```

### Root cause

- `outputs[0].try_extract_tensor::<f32>()` returns:
  - `(&Shape, &[f32])` (shape + raw data slice)
- You must build an `ndarray::ArrayViewD` manually from shape + slice.

### Fix (code change)

```rust
let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
let predictions = ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&dims), data)
    .expect("Invalid output shape");
```

---

## 5) Problem: Linker error from ORT static artifact (`__isoc23_*`)

### Symptom / Error (link step)

```
rust-lld: error: undefined symbol: __isoc23_strtoll
rust-lld: error: undefined symbol: __isoc23_strtol
rust-lld: error: undefined symbol: __isoc23_strtoull
```

### Root cause

This happens when **building/linking ORT (or parts of it) in a way that expects newer glibc symbols** (C23 versions), but the linker environment doesn’t provide them in the expected way.

### Fix strategy used

Instead of relying on whatever was being built/linked implicitly, we switched to loading a known good ONNX Runtime **dynamic library** (`libonnxruntime.so`) from a prebuilt distribution.

---

## 6) Problem: Ubuntu `apt` package `onnxruntime-gpu` not available

### Symptom

```bash
sudo apt-get install onnxruntime-gpu
# -> E: Unable to locate package onnxruntime-gpu
```

### Root cause

Ubuntu repositories typically do not ship a package named `onnxruntime-gpu` for Jammy in default repos.

### Fix

Use Python wheels as a known source of ONNX Runtime GPU binaries.

---

## 7) Install ONNX Runtime GPU wheel and locate the ORT shared library

### Commands

```bash
python3 -m pip install --user onnxruntime-gpu
```

Locate the ORT library:

```bash
python3 -c "import onnxruntime, os; import onnxruntime.capi._pybind_state as s; print(os.path.join(os.path.dirname(s.__file__), 'libonnxruntime.so'))"
```

### Observation

The wheel contains:

- `libonnxruntime.so.1.23.2`
- `libonnxruntime_providers_cuda.so`
- `libonnxruntime_providers_shared.so`
- (optionally) TensorRT provider `.so`

The actual core library file was **not** `libonnxruntime.so` but versioned:

```bash
ls -la ~/.local/lib/python3.10/site-packages/onnxruntime/capi/
# shows: libonnxruntime.so.1.23.2, libonnxruntime_providers_cuda.so, ...
```

---

## 8) Problem: ORT dylib path wrong → “No such file or directory”

### Symptom

- Set `ORT_DYLIB_PATH` to a non-existent file:

```bash
export ORT_DYLIB_PATH=~/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so
```

Then:

- ORT failed to load
- `ldd "$ORT_DYLIB_PATH"` reported “No such file or directory”

### Root cause

The wheel uses a versioned filename: `libonnxruntime.so.1.23.2`, not `libonnxruntime.so`.

### Fix

```bash
export ORT_DYLIB_PATH=~/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2
```

---

## 9) Problem: CUDA provider `.so` fails to load (missing CUDA libs)

### Symptom in the app

A diagnostic check printed:

```
CUDA provider .so loadable: NO ❌ (.../libonnxruntime_providers_cuda.so)
dlopen error: dlopen failed
```

### Confirm with `ldd`

```bash
ldd ~/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so \
  | grep -E "not found|cuda|cud|cublas|cudnn" || true
```

### Output showed missing runtime libs

```
libcublasLt.so.12 => not found
libcublas.so.12 => not found
libcufft.so.11 => not found
libcudart.so.12 => not found
libcudnn.so.9 => not found
```

### Root cause

The ORT GPU wheel expects CUDA runtime libraries to exist on the system (or in the runtime library path).

---

## 10) Attempted fix: install CUDA libs via `apt` (failed)

### Commands

```bash
sudo apt-get install -y \
  cuda-cudart-12-6 \
  libcublas-12-6 \
  libcufft-12-6
```

### Symptom

```
E: Unable to locate package cuda-cudart-12-6
E: Unable to locate package libcublas-12-6
E: Unable to locate package libcufft-12-6
```

### Root cause

Those exact package names are not present in the default Ubuntu repositories without adding NVIDIA’s CUDA apt repo.

---

## 11) Attempted fix: conda `cudatoolkit=12` (failed)

### Command

```bash
conda activate workenv
conda install -c conda-forge -y cudatoolkit=12 cudnn=9
```

### Symptom

```
PackagesNotFoundError: cudatoolkit=12
```

### Root cause

conda-forge has shifted packaging; `cudatoolkit` naming/pinning can differ. The correct packages are typically the modular `cuda-*` components or `cuda-toolkit`.

---

## 12) Fix: install `mamba` then install CUDA toolkit libs

### Install mamba

```bash
conda activate workenv
conda install -c conda-forge -y mamba
```

### Install CUDA toolkit + cuDNN (first pass pulled CUDA 13.1)

```bash
mamba install -c conda-forge -y cuda-toolkit cudnn=9
```

### Result

CUDA libs installed, but the ORT wheel was expecting **CUDA 12 era SONAMEs** (`libcudart.so.12`, `libcublas.so.12`, `libcufft.so.11`).
The first install ended up giving a CUDA 13.x set, creating a mismatch.

---

## 13) Fix: pin CUDA to 12.x + install matching libraries (this solved missing `.so`)

### Commands (the winning ones)

```bash
conda activate workenv

mamba install -c conda-forge -y "cuda-version=12.*" cudnn=9
mamba install -c conda-forge -y "libcublas=12.*" "cuda-cudart=12.*" "libcufft=11.*"
```

### Verify with `ldd` (after pinning)

```bash
ldd ~/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so \
  | grep -E "not found|cudart|cublas|cufft|cudnn" || true
```

### Successful output (no “not found”)

```
libcublasLt.so.12 => .../workenv/lib/libcublasLt.so.12
libcublas.so.12   => .../workenv/lib/libcublas.so.12
libcufft.so.11    => .../workenv/lib/libcufft.so.11
libcudart.so.12   => .../workenv/lib/libcudart.so.12
libcudnn.so.9     => .../workenv/lib/libcudnn.so.9
```

---

## 14) Problem: Provider `.so` dlopen fails with undefined symbol `Provider_GetHost`

### Symptom (after CUDA libs existed)

When checking CUDA provider loadability:

```
CUDA provider .so loadable: NO ❌ (.../libonnxruntime_providers_cuda.so)
dlopen error: ... undefined symbol: Provider_GetHost
```

### Root cause

This is a **symbol visibility / dynamic loader namespace** issue:

- The CUDA provider `.so` expects symbols from:
  - ORT core (`libonnxruntime.so.*`)
  - ORT shared provider library (`libonnxruntime_providers_shared.so`)
- If ORT core is not loaded into the **global** symbol namespace, a later `dlopen` on the CUDA provider may fail resolving those symbols.

In other words: the provider loads fine _only if_ the ORT core + shared provider libs are already loaded and globally visible.

---

## 15) Fix: preload ORT core + providers_shared with `RTLD_GLOBAL`

### What we did

We added a small Rust function that calls `dlopen(..., RTLD_NOW | RTLD_GLOBAL)` on:

1. `libonnxruntime.so.1.23.2`
2. `libonnxruntime_providers_shared.so`

### Runtime environment (important ordering)

```bash
conda activate workenv

export ORT_DYLIB_PATH=~/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2

# ORT dir first (find providers_shared), then conda (find cudart/cublas/cufft/cudnn)
export LD_LIBRARY_PATH=~/.local/lib/python3.10/site-packages/onnxruntime/capi:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Result

App prints:

```
Preloaded (GLOBAL): .../libonnxruntime.so.1.23.2
Preloaded (GLOBAL): .../libonnxruntime_providers_shared.so
Model loaded successfully!
```

At this point, CUDA provider registration works and ORT can use the GPU.

---

## 16) Final confirmation: GPU usage visible in `nvidia-smi`

### Command (run in another terminal while app runs)

```bash
watch -n 0.5 nvidia-smi
```

### Expected observation

- Your process appears under “Processes” (Type `C` compute):
  - `/pipeline`
- GPU memory usage is non-zero
- GPU utilization (%) changes over time during inference

Example observed:

```
Processes:
GPU   PID   Type   Process name
0   28720   C      /pipeline
```

This confirms the inference pipeline is **actually executing on GPU**.

---

# Appendix A — Key environment exports (final working set)

```bash
conda activate workenv

export ORT_DYLIB_PATH=/home/amirali/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2
export LD_LIBRARY_PATH=/home/amirali/.local/lib/python3.10/site-packages/onnxruntime/capi:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

# Appendix B — Why we pinned CUDA to 12.x

The ORT GPU wheel (`onnxruntime-gpu 1.23.2`) pulled provider binaries built against a CUDA 12 era ABI.
That means the provider expects these SONAMEs:

- `libcudart.so.12`
- `libcublas.so.12`
- `libcublasLt.so.12`
- `libcufft.so.11`
- `libcudnn.so.9`

Installing CUDA 13.x gives different SONAME versions (e.g., cuBLAS 13, cuFFT 12), which **won’t satisfy** the provider loader, so we pinned CUDA to `12.*`.

---

# Appendix C — The “glibc \__isoc23_\*” linker error (what it meant)

When you saw:

- `undefined symbol: __isoc23_strtol`, etc.

it indicated the linked ORT artifacts in that build path expected newer C-library symbols. Switching to the **wheel’s prebuilt** shared library and loading it via `ORT_DYLIB_PATH` avoided that build/link mismatch in your Rust binary.

---

# Appendix D — Practical “make it reproducible” suggestion

Create a small script `run_gpu.sh` (optional) that sets env vars and runs:

```bash
#!/usr/bin/env bash
set -euo pipefail

conda activate workenv

export ORT_DYLIB_PATH="$HOME/.local/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.10/site-packages/onnxruntime/capi:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cargo run
```

---

## End state

✅ Project builds on `ort 2.0.0-rc.11`
✅ CUDA provider loads correctly
✅ `nvidia-smi` shows `/pipeline` as a GPU compute process
✅ GPU utilization changes during inference
