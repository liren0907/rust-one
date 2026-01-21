//! # 資料科學與統計教學程式庫
//!
//! 這個程式庫提供實用的資料科學和統計學習工具，包含：
//! - 基礎統計運算 (均值、中位數、眾數、變異數、標準差)
//! - 資料結構 (Series, DataFrame)
//! - 常見演算法 (線性回歸、相關係數)
//! - 機率分布
//! - 假設檢定
//!
//! ## 使用範例
//!
//! ```rust
//! use data_science::*;
//!
//! // 計算基礎統計量
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let stats = BasicStats::new(&data);
//!
//! println!("平均值: {}", stats.mean());
//! println!("標準差: {}", stats.std_dev());
//! ```

pub mod stats;
pub mod data_structures;
pub mod algorithms;
pub mod distributions;
pub mod testing;

// 重新匯出主要功能
pub use stats::*;
pub use data_structures::*;
pub use algorithms::*;
pub use distributions::*;
pub use testing::*;
