use std::ffi::{CStr, c_char, c_int};

// Declare C++ functions
#[link(name = "math", kind = "static")]
unsafe extern "C" {
    fn calculate_pi(digits: c_int) -> *mut c_char;
    fn free_pi_string(pi_str: *mut c_char);
}

fn main() {
    println!("🔗 C++ Binding Tutorial");
    println!("---------------------");

    let desired_digits = 50;

    unsafe {
        // 1. Call C++ function
        let pi_c_ptr = calculate_pi(desired_digits);

        if pi_c_ptr.is_null() {
            eprintln!("Error: C++ returned null pointer");
            return;
        }

        // 2. Convert C string to Rust string
        let pi_c_str = CStr::from_ptr(pi_c_ptr as *const c_char);
        match pi_c_str.to_str() {
            Ok(pi_str) => {
                println!("Pi ({} digits): {}", desired_digits, pi_str);
            }
            Err(e) => {
                eprintln!("Conversion error: {}", e);
            }
        }

        // 3. Free memory allocated by C++
        free_pi_string(pi_c_ptr);
        println!("Memory freed successfully.");
    }
}
