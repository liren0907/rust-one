//! 機器學習演算法示範
//!
//! 展示如何使用 data-science-tutorial crate 實作常見的機器學習演算法

use data_science::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 機器學習演算法示範");
    println!("===================\n");

    // 1. KNN 分類器
    knn_demo()?;

    // 2. 決策樹分類器
    decision_tree_demo()?;

    // 3. 模型比較
    model_comparison_demo()?;

    println!("\n🎯 機器學習示範完成！");
    Ok(())
}

/// KNN 分類器示範
fn knn_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 K-最近鄰 (KNN) 分類器");
    println!("--------------------");

    // 建立訓練資料：鳶尾花資料集簡化版
    // 特徵：[花萼長度, 花萼寬度, 花瓣長度, 花瓣寬度]
    let training_data = vec![
        vec![5.1, 3.5, 1.4, 0.2], // Setosa
        vec![4.9, 3.0, 1.4, 0.2], // Setosa
        vec![4.7, 3.2, 1.3, 0.2], // Setosa
        vec![7.0, 3.2, 4.7, 1.4], // Versicolor
        vec![6.4, 3.2, 4.5, 1.5], // Versicolor
        vec![6.9, 3.1, 4.9, 1.5], // Versicolor
        vec![6.3, 3.3, 6.0, 2.5], // Virginica
        vec![5.8, 2.7, 5.1, 1.9], // Virginica
        vec![7.1, 3.0, 5.9, 2.1], // Virginica
    ];

    let labels = vec![
        "Setosa".to_string(),
        "Setosa".to_string(),
        "Setosa".to_string(),
        "Versicolor".to_string(),
        "Versicolor".to_string(),
        "Versicolor".to_string(),
        "Virginica".to_string(),
        "Virginica".to_string(),
        "Virginica".to_string(),
    ];

    // 訓練 KNN 分類器 (K=3)
    let mut knn = KNNClassifier::new(3);
    knn.fit(training_data, labels)?;

    // 測試資料
    let test_samples = vec![
        vec![5.0, 3.4, 1.5, 0.2], // 應該是 Setosa
        vec![6.5, 3.0, 5.2, 2.0], // 應該是 Virginica
        vec![6.0, 2.9, 4.5, 1.5], // 應該是 Versicolor
    ];

    println!("KNN 分類結果 (K=3):");
    for (i, sample) in test_samples.iter().enumerate() {
        let prediction = knn.predict(sample)?;
        println!("樣本 {}: 預測為 {}", i + 1, prediction);
    }

    // 測試不同的 K 值
    println!("\n不同 K 值的比較:");
    let test_sample = vec![6.1, 2.8, 4.7, 1.2]; // Versicolor

    for k in [1, 3, 5] {
        let mut knn_temp = KNNClassifier::new(k);
        knn_temp.fit(knn.training_data().to_vec(), knn.labels().to_vec())?;
        let prediction = knn_temp.predict(&test_sample)?;
        println!("K={}, 預測: {}", k, prediction);
    }
    println!();

    Ok(())
}

/// 決策樹分類器示範
fn decision_tree_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌳 決策樹分類器");
    println!("-------------");

    // 簡化的二元分類問題
    // 特徵：[年齡, 收入等級(1=低, 2=中, 3=高), 信用評分]
    let training_data = vec![
        vec![25.0, 1.0, 650.0], // 不批准
        vec![35.0, 2.0, 700.0], // 批准
        vec![45.0, 3.0, 800.0], // 批准
        vec![30.0, 1.0, 600.0], // 不批准
        vec![40.0, 2.0, 750.0], // 批准
        vec![50.0, 3.0, 850.0], // 批准
        vec![28.0, 1.0, 620.0], // 不批准
        vec![38.0, 2.0, 720.0], // 批准
    ];

    let labels = vec![
        "不批准".to_string(),
        "批准".to_string(),
        "批准".to_string(),
        "不批准".to_string(),
        "批准".to_string(),
        "批准".to_string(),
        "不批准".to_string(),
        "批准".to_string(),
    ];

    // 訓練決策樹
    let mut tree = SimpleDecisionTree::new();
    tree.fit(&training_data, &labels, 3)?;

    // 測試樣本
    let test_samples = vec![
        vec![32.0, 2.0, 680.0], // 應該批准
        vec![26.0, 1.0, 580.0], // 應該不批准
        vec![42.0, 3.0, 780.0], // 應該批准
    ];

    println!("決策樹分類結果:");
    for (i, sample) in test_samples.iter().enumerate() {
        if let Some(prediction) = tree.predict(sample) {
            println!("樣本 {}: 預測為 {}", i + 1, prediction);
        }
    }
    println!();

    Ok(())
}

/// 模型比較示範
fn model_comparison_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️ 模型比較");
    println!("---------");

    // 準備簡單的二元分類資料
    let features = vec![
        vec![2.0, 3.0],
        vec![3.0, 4.0],
        vec![4.0, 5.0],
        vec![5.0, 6.0],
        vec![6.0, 7.0],
        vec![7.0, 8.0],
        vec![8.0, 9.0],
        vec![9.0, 10.0],
    ];

    let labels = vec![
        "A".to_string(),
        "A".to_string(),
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string(),
        "B".to_string(),
        "B".to_string(),
    ];

    // 訓練 KNN 模型 (不同 K 值)
    let mut knn_results = HashMap::new();
    for k in [1, 3, 5] {
        let mut knn = KNNClassifier::new(k);
        knn.fit(features.clone(), labels.clone())?;

        // 簡單的交叉驗證 (這裡用訓練資料測試，實際應用中應該用驗證資料)
        let mut correct = 0;
        for i in 0..features.len() {
            if let Ok(prediction) = knn.predict(&features[i]) {
                if prediction == labels[i] {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f64 / features.len() as f64;
        knn_results.insert(k, accuracy);
    }

    println!("KNN 模型準確率比較:");
    for (k, accuracy) in &knn_results {
        println!("K={}: {:.1}%", k, accuracy * 100.0);
    }

    // 訓練決策樹模型
    let mut tree = SimpleDecisionTree::new();
    tree.fit(&features, &labels, 3)?;

    let mut tree_correct = 0;
    for i in 0..features.len() {
        if let Some(prediction) = tree.predict(&features[i]) {
            if prediction == labels[i] {
                tree_correct += 1;
            }
        }
    }

    let tree_accuracy = tree_correct as f64 / features.len() as f64;
    println!("決策樹準確率: {:.1}%", tree_accuracy * 100.0);

    // 比較結果
    println!("\n模型比較總結:");
    let best_knn = knn_results
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("最佳 KNN (K={}): {:.1}%", best_knn.0, best_knn.1 * 100.0);
    println!("決策樹: {:.1}%", tree_accuracy * 100.0);

    if *best_knn.1 > tree_accuracy {
        println!("🏆 KNN 表現較佳");
    } else if tree_accuracy > *best_knn.1 {
        println!("🏆 決策樹表現較佳");
    } else {
        println!("🤝 兩個模型表現相當");
    }
    println!();

    Ok(())
}
