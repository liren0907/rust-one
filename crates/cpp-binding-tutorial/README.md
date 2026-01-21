# C++ Binding Tutorial

This crate demonstrates how to call C++ code from Rust using Foreign Function Interface (FFI). It shows a practical example of integrating a legacy or high-performance C++ library into a Rust application.

## 📋 Overview

The project calculates the value of Pi to a specified number of digits.
- **core logic**: Implemented in **C++** (string manipulation and memory allocation).
- **Control**: Managed by **Rust** (calling the function, printing results, and ensuring memory is freed).

## 🛠️ Prerequisites

To run this tutorial, you need:
- **Rust toolchain** (cargo, rustc)
- **C++ compiler** (e.g., `g++`, `clang++`, or MSVC on Windows)
- **`cc` crate**: Automatically handles C++ compilation in `build.rs`.

## 📂 Project Structure

```text
.
├── Cargo.toml      # Dependencies (includes `cc` for build)
├── build.rs        # Build script to compile C++ code
└── src/
    ├── main.rs     # Rust entry point (FFI definitions)
    └── cpp/
        ├── math.cpp    # C++ implementation
        └── math.h      # C Header for binding
```

## 🚀 How It Works

### 1. The C++ Side (`src/cpp/math.cpp`)
Rust cannot call C++ classes or templates directly due to name mangling. To expose C++ functions to Rust, we use `extern "C"` to create a C-compatible interface.

```cpp
extern "C" {
    // Calculates Pi and returns a pointer to a C-string (char*)
    char *calculate_pi(int digits) {
        // ... allocates memory using malloc ...
        return c_str;
    }

    // Frees the memory allocated by calculate_pi
    void free_pi_string(char *pi_str) {
        free(pi_str);
    }
}
```

### 2. The Build Script (`build.rs`)
Before Rust code is compiled, `build.rs` runs. It uses the `cc` crate to compile the C++ source into a static library (`libmath.a` or `math.lib`).

```rust
fn main() {
    cc::Build::new()
        .cpp(true)               // Enable C++ support
        .file("src/cpp/math.cpp") // Source file
        .compile("math");        // Output library name
}
```

### 3. The Rust Side (`src/main.rs`)
Rust defines the external functions within an `extern "C"` block.

```rust
#[link(name = "math", kind = "static")]
unsafe extern "C" {
    fn calculate_pi(digits: c_int) -> *mut c_char;
    fn free_pi_string(pi_str: *mut c_char);
}
```

Calls to foreign functions are always `unsafe` because the Rust compiler cannot guarantee memory safety or thread safety for external code.

## 💡 Important FFI Tips

### Name Mangling
Standard C++ functions mangle their names (e.g., `_Z12calculate_pi`).
- **Solution**: Always wrap functions meant for FFI in `extern "C" { ... }` in your C++ code.

### Memory Management
**Rule of Thumb**: "Who allocates, must free."
- If C++ converts a `malloc`'d pointer to Rust, Rust relies on C++ to `free` it.
- In this example, `calculate_pi` allocates memory, so we provide `free_pi_string` to deallocate it properly.
- **Do not** try to drop a C pointer in Rust using Rust's allocator (System or Jemalloc); it may cause a segfault.

### Type Compatibility
Ensure types match between languages:
- `int` (C++)  ↔ `c_int` (Rust)
- `char*` (C++) ↔ `*mut c_char` (Rust)
- `void` (C++)  ↔ `()` (Rust)
