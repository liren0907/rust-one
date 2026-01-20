fn main() {
    println!("cargo:rerun-if-changed=src/cpp/math.cpp");
    println!("cargo:rerun-if-changed=src/cpp/math.h");

    cc::Build::new()
        .cpp(true)
        .file("src/cpp/math.cpp")
        .compile("math");
}
