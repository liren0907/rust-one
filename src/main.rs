//! 資料科學與統計教學示範程式
//!
//! 這個程式展示如何使用 data_science crate 進行資料分析

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 資料科學與統計教學示範");
    println!("==========================\n");

    println!("💡 提示: 執行以下指令查看詳細示範：");
    println!("   cargo run -p data-science --example data_science_demo");
    println!("   cargo run -p data-science --example statistics_tutorial");
    println!("   cargo run -p data-science --example machine_learning_demo");

    println!("\n🎉 資料科學教學示範完成！");
    Ok(())
}
