//! 統計學實戰教學
//!
//! 通過實際案例展示統計分析的應用

use data_science::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 統計學實戰教學");
    println!("================\n");

    // 1. 學生成績分析
    student_performance_analysis()?;

    // 2. A/B 測試分析
    ab_test_analysis()?;

    // 3. 品質控制分析
    quality_control_analysis()?;

    // 4. 市場調查分析
    market_research_analysis()?;

    println!("\n📊 統計教學完成！");
    Ok(())
}

/// 學生成績分析案例
fn student_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 案例 1: 學生成績分析");
    println!("-------------------");

    // 模擬學生成績資料
    let math_scores = vec![85.0, 92.0, 78.0, 96.0, 88.0, 91.0, 83.0, 89.0, 94.0, 87.0];
    let science_scores = vec![82.0, 88.0, 91.0, 87.0, 93.0, 85.0, 89.0, 92.0, 86.0, 90.0];
    let english_scores = vec![78.0, 85.0, 88.0, 82.0, 91.0, 87.0, 83.0, 89.0, 84.0, 86.0];

    // 分析每門科目的成績分布
    let subjects = vec![
        ("數學", &math_scores),
        ("自然", &science_scores),
        ("英文", &english_scores),
    ];

    println!("各科目成績統計摘要:");
    for (subject, scores) in &subjects {
        let stats = BasicStats::new(scores);
        println!("\n{} 成績:", subject);
        println!("  平均: {:.1}", stats.mean());
        println!("  標準差: {:.1}", stats.std_dev());
        println!("  最高: {:.1}", stats.sorted_data().last().unwrap());
        println!("  最低: {:.1}", stats.sorted_data().first().unwrap());
        println!("  中位數: {:.1}", stats.median());
    }

    // 計算科目間的相關性
    println!("\n科目間相關係數:");
    if let Ok(corr_ms) = correlation(&math_scores, &science_scores) {
        println!("數學 vs 自然: {:.3}", corr_ms);
    }
    if let Ok(corr_me) = correlation(&math_scores, &english_scores) {
        println!("數學 vs 英文: {:.3}", corr_me);
    }
    if let Ok(corr_se) = correlation(&science_scores, &english_scores) {
        println!("自然 vs 英文: {:.3}", corr_se);
    }

    // 成績等第分析
    println!("\n成績等第分布:");
    for (subject, scores) in &subjects {
        let grades = scores
            .iter()
            .map(|&score| match score as u32 {
                90..=100 => "A",
                80..=89 => "B",
                70..=79 => "C",
                60..=69 => "D",
                _ => "F",
            })
            .collect::<Vec<&str>>();

        let mut grade_counts = HashMap::new();
        for &grade in &grades {
            *grade_counts.entry(grade).or_insert(0) += 1;
        }

        println!("\n{} 等第分布:", subject);
        for grade in ["A", "B", "C", "D", "F"] {
            let count = grade_counts.get(grade).unwrap_or(&0);
            let percentage = (*count as f64 / grades.len() as f64) * 100.0;
            println!("  {}: {} 人 ({:.1}%)", grade, count, percentage);
        }
    }
    println!();

    Ok(())
}

/// A/B 測試分析案例
fn ab_test_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("🅰️🅱️ 案例 2: A/B 測試分析");
    println!("-----------------");

    // 模擬 A/B 測試資料：按鈕顏色對轉換率的影響
    let group_a_conversions = vec![24, 28, 32, 26, 30, 27, 31, 29, 25, 33]; // 藍色按鈕
    let group_b_conversions = vec![31, 35, 38, 33, 36, 34, 39, 37, 32, 40]; // 紅色按鈕

    let n_a = 200; // A組總樣本數
    let n_b = 200; // B組總樣本數

    // 轉換率計算
    let rate_a = group_a_conversions.iter().sum::<i32>() as f64 / n_a as f64;
    let rate_b = group_b_conversions.iter().sum::<i32>() as f64 / n_b as f64;

    println!("A組 (藍色按鈕): 轉換率 = {:.1}%", rate_a * 100.0);
    println!("B組 (紅色按鈕): 轉換率 = {:.1}%", rate_b * 100.0);
    println!("轉換率差異: {:.1}%", (rate_b - rate_a) * 100.0);

    // 進行比例檢定
    let total_a = group_a_conversions.iter().sum::<i32>() as usize;
    let total_b = group_b_conversions.iter().sum::<i32>() as usize;

    let pooled_rate = (total_a + total_b) as f64 / (n_a + n_b) as f64;
    let test_result = one_sample_proportion_test(total_b, n_b, pooled_rate, 0.05)?;

    println!("\n統計檢定結果:");
    println!("{}", test_result.summary());

    // 計算信賴區間
    let se = (rate_a * (1.0 - rate_a) / n_a as f64 + rate_b * (1.0 - rate_b) / n_b as f64).sqrt();
    let z = 1.96; // 95% 信賴區間
    let ci_lower = (rate_b - rate_a) - z * se;
    let ci_upper = (rate_b - rate_a) + z * se;

    println!(
        "轉換率差異的 95% 信賴區間: [{:.1}%, {:.1}%]",
        ci_lower * 100.0,
        ci_upper * 100.0
    );

    if ci_lower > 0.0 {
        println!("🎯 結論: B組表現顯著優於 A組");
    } else if ci_upper < 0.0 {
        println!("🎯 結論: A組表現顯著優於 B組");
    } else {
        println!("🤔 結論: 兩組間沒有顯著差異");
    }
    println!();

    Ok(())
}

/// 品質控制分析案例
fn quality_control_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭 案例 3: 品質控制分析");
    println!("-----------------");

    // 模擬產品重量資料 (目標重量: 100g, 容許誤差: ±5g)
    let weights = vec![
        98.5, 101.2, 99.8, 100.5, 102.1, 97.8, 100.9, 101.8, 99.2, 100.1, 98.9, 101.5, 100.3, 99.7,
        101.0, 98.4, 100.7, 99.5, 101.3, 100.2, 99.1, 100.8, 98.7, 101.7, 100.4, 99.9, 101.4, 98.3,
        100.6, 99.3,
    ];

    let target_weight = 100.0;
    let tolerance = 5.0;

    // 基本統計分析
    let stats = BasicStats::new(&weights);
    println!("產品重量統計:");
    println!("樣本數: {}", weights.len());
    println!("平均重量: {:.2}g", stats.mean());
    println!("標準差: {:.2}g", stats.std_dev());
    println!("變異係數: {:.1}%", (stats.std_dev() / stats.mean()) * 100.0);

    // 品質分析
    let defects = weights
        .iter()
        .filter(|&&w| (w - target_weight).abs() > tolerance)
        .count();

    let defect_rate = defects as f64 / weights.len() as f64;
    println!("不合格品數: {} 個", defects);
    println!("不良率: {:.1}%", defect_rate * 100.0);

    // 控制圖分析 (簡化版本)
    let ucl = stats.mean() + 3.0 * stats.std_dev(); // 上控制限
    let lcl = stats.mean() - 3.0 * stats.std_dev(); // 下控制限

    println!("控制圖分析:");
    println!("中心線 (CL): {:.2}g", stats.mean());
    println!("上控制限 (UCL): {:.2}g", ucl);
    println!("下控制限 (LCL): {:.2}g", lcl);

    let out_of_control = weights.iter().filter(|&&w| w > ucl || w < lcl).count();

    println!("超出控制限的樣本數: {}", out_of_control);

    if out_of_control == 0 {
        println!("✅ 製程處於統計控制狀態");
    } else {
        println!("⚠️  製程可能有異常，需要進一步調查");
    }

    // 能力分析
    let cp = (2.0 * tolerance) / (6.0 * stats.std_dev()); // 製程能力指數
    let cpk = ((target_weight - stats.mean()).abs() / (3.0 * stats.std_dev())).min(cp); // 製程能力指數 (考量偏移)

    println!("\n製程能力分析:");
    println!("Cp: {:.3}", cp);
    println!("Cpk: {:.3}", cpk);

    match cpk {
        x if x >= 1.33 => println!("🏆 製程能力優良"),
        x if x >= 1.0 => println!("✅ 製程能力合格"),
        _ => println!("⚠️  製程能力不足"),
    }
    println!();

    Ok(())
}

/// 市場調查分析案例
fn market_research_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 案例 4: 市場調查分析");
    println!("-----------------");

    // 模擬客戶滿意度調查 (1-5 分制)
    let satisfaction_scores = vec![
        5, 4, 5, 3, 4, 5, 4, 3, 5, 4, // 產品A
        4, 3, 4, 5, 4, 3, 4, 5, 4, 3, // 產品B
        3, 4, 3, 4, 5, 3, 4, 3, 4, 5, // 產品C
    ];

    let products = vec!["產品A", "產品B", "產品C"];
    let chunk_size = 10;

    println!("客戶滿意度分析:");

    for (i, product) in products.iter().enumerate() {
        let start = i * chunk_size;
        let end = start + chunk_size;
        let product_scores: Vec<f64> = satisfaction_scores[start..end]
            .iter()
            .map(|&x| x as f64)
            .collect();

        let stats = BasicStats::new(&product_scores);
        let mean_score = stats.mean();
        let satisfaction_rate = product_scores.iter().filter(|&&x| x >= 4.0).count() as f64
            / product_scores.len() as f64;

        println!("\n{}:", product);
        println!("  平均分數: {:.2}/5.0", mean_score);
        println!("  滿意度比例 (≥4分): {:.1}%", satisfaction_rate * 100.0);
        println!("  得分分布: {:?}", product_scores);
    }

    // 卡方檢定：檢查產品間滿意度分布是否有顯著差異
    let contingency_table = vec![
        vec![7, 3], // 產品A: [滿意(4-5), 不滿意(1-3)]
        vec![6, 4], // 產品B: [滿意(4-5), 不滿意(1-3)]
        vec![6, 4], // 產品C: [滿意(4-5), 不滿意(1-3)]
    ];

    let chi_test = chi_square_independence(&contingency_table, 0.05)?;
    println!("\n產品間滿意度差異檢定:");
    println!("{}", chi_test.summary());

    // 建議
    println!("\n📋 市場建議:");
    if chi_test.reject_null {
        println!("• 不同產品的滿意度分布有顯著差異");
        println!("• 建議深入分析各產品的優勢和劣勢");
        println!("• 可以考慮針對不滿意客戶提供改進措施");
    } else {
        println!("• 各產品滿意度分布沒有顯著差異");
        println!("• 客戶對各產品的滿意度相當一致");
        println!("• 可以統一客戶服務策略");
    }
    println!();

    Ok(())
}
