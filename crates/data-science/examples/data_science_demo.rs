//! 資料科學教學示範程式
//!
//! 這個範例展示了如何使用 data-science-tutorial crate 進行資料分析

use data_science::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 資料科學與統計教學示範");
    println!("==========================\n");

    // 1. 基礎統計分析
    basic_statistics_demo()?;

    // 2. 資料結構操作
    data_structures_demo()?;

    // 3. 線性回歸分析
    linear_regression_demo()?;

    // 4. 機率分布
    distributions_demo()?;

    // 5. 假設檢定
    hypothesis_testing_demo()?;

    println!("\n🎉 資料科學教學示範完成！");
    Ok(())
}

/// 基礎統計分析示範
fn basic_statistics_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 基礎統計分析");
    println!("-------------");

    // 測試資料
    let data = vec![85.0, 92.0, 78.0, 96.0, 88.0, 91.0, 83.0, 89.0, 94.0, 87.0];

    let stats = BasicStats::new(&data);

    println!("原始資料: {:?}", data);
    println!("平均值: {:.2}", stats.mean());
    println!("中位數: {:.2}", stats.median());
    println!("眾數: {:?}", stats.mode());
    println!("變異數: {:.2}", stats.variance());
    println!("標準差: {:.2}", stats.std_dev());
    println!("範圍: {:.2}", stats.range());
    println!("四分位距: {:.2}", stats.iqr());
    println!("偏態係數: {:.2}", stats.skewness());
    println!("峰態係數: {:.2}", stats.kurtosis());
    println!();

    // 百分位數計算
    for p in [25.0, 50.0, 75.0, 95.0] {
        let percentile = percentile(&data, p)?;
        println!("第 {:.0} 百分位數: {:.2}", p, percentile);
    }
    println!();

    Ok(())
}

/// 資料結構操作示範
fn data_structures_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 資料結構操作");
    println!("-------------");

    // 建立 Series
    let scores = vec![85.0, 92.0, 78.0, 96.0, 88.0];
    let series = Series::new("成績", scores);

    println!("Series: {}", series.name());
    println!("長度: {}", series.len());
    println!("資料: {:?}", series.data());

    // Series 統計資訊
    let stats = series.describe();
    println!("\n統計摘要:");
    println!("{}", stats.format());
    println!();

    // 建立 DataFrame
    let mut data = HashMap::new();
    data.insert("數學".to_string(), vec![85.0, 92.0, 78.0, 96.0, 88.0]);
    data.insert("英文".to_string(), vec![82.0, 88.0, 91.0, 87.0, 93.0]);
    data.insert("物理".to_string(), vec![88.0, 85.0, 92.0, 89.0, 90.0]);

    let df = DataFrame::from_hashmap(data)?;
    println!("DataFrame 欄位: {:?}", df.columns());
    println!(
        "DataFrame 形狀: {} 列 x {} 欄",
        df.len(),
        df.columns().len()
    );

    // 顯示前幾行
    println!("\nDataFrame 內容:");
    println!("{}", df.head(5));

    // 計算各欄位統計資訊
    let df_stats = df.describe();
    println!("\n各科目統計摘要:");
    for (col, stats) in df_stats {
        println!("{}: 平均={:.2}, 標準差={:.2}", col, stats.mean, stats.std);
    }

    // 計算相關係數矩陣
    let correlations = df.correlation_matrix();
    println!("\n相關係數矩陣:");
    for ((col1, col2), corr) in correlations {
        if col1 != col2 {
            println!("{} vs {}: {:.3}", col1, col2, corr);
        }
    }
    println!();

    Ok(())
}

/// 線性回歸分析示範
fn linear_regression_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 線性回歸分析");
    println!("-------------");

    // 簡單的線性回歸範例：學習時間 vs 成績
    let study_hours = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let scores = vec![60.0, 65.0, 70.0, 75.0, 78.0, 82.0, 85.0, 88.0, 90.0, 92.0];

    // 擬合線性回歸模型
    let model = LinearRegression::fit(&study_hours, &scores)?;
    println!("線性回歸模型摘要:");
    println!("{}", model.summary());

    // 預測
    let test_hours = vec![2.5, 6.5, 11.0];
    println!("\n預測結果:");
    for hours in test_hours {
        let predicted = model.predict_single(hours);
        println!("學習 {:.1} 小時預測成績: {:.1}", hours, predicted);
    }

    // 多項式回歸範例
    let poly_model = PolynomialRegression::fit(&study_hours, &scores, 2)?;
    println!("\n二階多項式回歸模型:");
    println!("{}", poly_model.summary());
    println!();

    Ok(())
}

/// 機率分布示範
fn distributions_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎲 機率分布");
    println!("---------");

    // 正態分布
    let normal = NormalDistribution::new(75.0, 10.0)?;
    println!("正態分布 N(μ=75, σ=10):");
    println!("PDF at x=75: {:.4}", normal.pdf(75.0));
    println!("CDF at x=85: {:.4}", normal.cdf(85.0));
    println!("分位數 p=0.95: {:.2}", normal.quantile(0.95));
    println!();

    // 二項分布
    let binomial = BinomialDistribution::new(10, 0.3)?;
    println!("二項分布 B(n=10, p=0.3):");
    println!("P(X=3): {:.4}", binomial.pmf(3));
    println!("P(X≤5): {:.4}", binomial.cdf(5));
    println!("期望值: {:.2}", binomial.mean());
    println!("變異數: {:.2}", binomial.variance());
    println!();

    // 泊松分布
    let poisson = PoissonDistribution::new(2.5)?;
    println!("泊松分布 Poisson(λ=2.5):");
    println!("P(X=2): {:.4}", poisson.pmf(2));
    println!("P(X≤3): {:.4}", poisson.cdf(3));
    println!("期望值: {:.2}", poisson.mean());
    println!();

    // 指數分布
    let exponential = ExponentialDistribution::from_mean(5.0)?;
    println!("指數分布 Exp(μ=5):");
    println!("PDF at x=2: {:.4}", exponential.pdf(2.0));
    println!("CDF at x=5: {:.4}", exponential.cdf(5.0));
    println!("生存函數 at x=5: {:.4}", exponential.survival(5.0));
    println!();

    Ok(())
}

/// 假設檢定示範
fn hypothesis_testing_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 假設檢定");
    println!("---------");

    // 單一樣本 t 檢定
    let sample_scores = vec![78.0, 82.0, 85.0, 79.0, 88.0, 84.0, 81.0, 87.0, 83.0, 86.0];
    let result = one_sample_t_test(&sample_scores, 80.0, 0.05)?;
    println!("單一樣本 t 檢定 (H₀: μ = 80):");
    println!("{}", result.summary());
    println!();

    // 獨立樣本 t 檢定
    let class_a = vec![85.0, 88.0, 82.0, 90.0, 87.0];
    let class_b = vec![78.0, 80.0, 75.0, 82.0, 79.0];
    let result = independent_t_test(&class_a, &class_b, 0.05)?;
    println!("獨立樣本 t 檢定 (A班 vs B班):");
    println!("{}", result.summary());
    println!();

    // 配對樣本 t 檢定
    let before_training = vec![70.0, 75.0, 68.0, 72.0, 69.0];
    let after_training = vec![78.0, 82.0, 75.0, 79.0, 76.0];
    let result = paired_t_test(&before_training, &after_training, 0.05)?;
    println!("配對樣本 t 檢定 (訓練前後比較):");
    println!("{}", result.summary());
    println!();

    // 比例檢定
    let result = one_sample_proportion_test(65, 100, 0.6, 0.05)?;
    println!("比例檢定 (H₀: p = 0.6):");
    println!("{}", result.summary());
    println!();

    Ok(())
}
