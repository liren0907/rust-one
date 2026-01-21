//! 資料科學演算法模組
//!
//! 提供常見的資料科學和機器學習演算法實作

/// 線性回歸模型
#[derive(Debug, Clone)]
pub struct LinearRegression {
    slope: f64,
    intercept: f64,
    r_squared: f64,
}

impl LinearRegression {
    /// 使用最小二乘法擬合線性回歸模型
    pub fn fit(x: &[f64], y: &[f64]) -> Result<Self, &'static str> {
        if x.len() != y.len() {
            return Err("自變數和應變數長度不一致");
        }

        if x.len() < 2 {
            return Err("資料點數量太少，無法進行回歸分析");
        }

        let n = x.len() as f64;
        let x_mean = x.iter().sum::<f64>() / n;
        let y_mean = y.iter().sum::<f64>() / n;

        // 計算斜率 (slope)
        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let denominator: f64 = x.iter()
            .map(|&xi| (xi - x_mean).powi(2))
            .sum();

        if denominator == 0.0 {
            return Err("自變數沒有變異性，無法計算斜率");
        }

        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;

        // 計算決定係數 R²
        let y_pred: Vec<f64> = x.iter().map(|&xi| slope * xi + intercept).collect();
        let ss_res: f64 = y.iter().zip(y_pred.iter())
            .map(|(&yi, &y_pred_i)| (yi - y_pred_i).powi(2))
            .sum();

        let ss_tot: f64 = y.iter()
            .map(|&yi| (yi - y_mean).powi(2))
            .sum();

        let r_squared = if ss_tot == 0.0 { 0.0 } else { 1.0 - (ss_res / ss_tot) };

        Ok(Self {
            slope,
            intercept,
            r_squared,
        })
    }

    /// 預測新資料點
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.slope * xi + self.intercept).collect()
    }

    /// 預測單一資料點
    pub fn predict_single(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    /// 取得斜率
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// 取得截距
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// 取得決定係數 R²
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 格式化輸出模型資訊
    pub fn summary(&self) -> String {
        format!(
            "線性回歸模型\n================\n方程式: y = {:.4}x + {:.4}\n決定係數 R²: {:.4}\n斜率: {:.4}\n截距: {:.4}",
            self.slope, self.intercept, self.r_squared, self.slope, self.intercept
        )
    }
}

/// 多項式回歸模型
#[derive(Debug, Clone)]
pub struct PolynomialRegression {
    degree: usize,
    coefficients: Vec<f64>,
    r_squared: f64,
}

impl PolynomialRegression {
    /// 擬合多項式回歸模型
    pub fn fit(x: &[f64], y: &[f64], degree: usize) -> Result<Self, &'static str> {
        if x.len() != y.len() {
            return Err("自變數和應變數長度不一致");
        }

        if x.len() < degree + 1 {
            return Err("資料點數量不足以擬合指定階數的多項式");
        }

        // 使用最小二乘法擬合多項式
        // 這裡使用簡單的矩陣求解，實際應用中可能需要更穩定的方法
        let n = x.len();
        let mut a = vec![vec![0.0; degree + 1]; degree + 1];
        let mut b = vec![0.0; degree + 1];

        // 建立法方程式
        for i in 0..=degree {
            for j in 0..=degree {
                a[i][j] = x.iter().map(|&xi| xi.powi((i + j) as i32)).sum();
            }
            b[i] = x.iter().zip(y.iter())
                .map(|(&xi, &yi)| yi * xi.powi(i as i32))
                .sum();
        }

        // 解聯立方程組 (這裡使用簡單的高斯消去法)
        let coefficients = Self::solve_linear_system(a, b)?;

        // 計算 R²
        let y_pred = Self::predict_with_coeffs(x, &coefficients);
        let y_mean = y.iter().sum::<f64>() / n as f64;

        let ss_res: f64 = y.iter().zip(y_pred.iter())
            .map(|(&yi, &y_pred_i)| (yi - y_pred_i).powi(2))
            .sum();

        let ss_tot: f64 = y.iter()
            .map(|&yi| (yi - y_mean).powi(2))
            .sum();

        let r_squared = if ss_tot == 0.0 { 0.0 } else { 1.0 - (ss_res / ss_tot) };

        Ok(Self {
            degree,
            coefficients,
            r_squared,
        })
    }

    /// 解聯立方程組 (高斯消去法)
    fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, &'static str> {
        let n = a.len();

        // 前進消去
        for i in 0..n {
            // 找主元素
            let mut max_row = i;
            for k in i + 1..n {
                if a[k][i].abs() > a[max_row][i].abs() {
                    max_row = k;
                }
            }

            // 交換行
            if max_row != i {
                a.swap(i, max_row);
                b.swap(i, max_row);
            }

            // 檢查主元素是否為零
            if a[i][i].abs() < 1e-10 {
                return Err("矩陣是奇異的，無法求解");
            }

            // 消去
            for k in i + 1..n {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
                b[k] -= factor * b[i];
            }
        }

        // 回代
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = b[i];
            for j in i + 1..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        Ok(x)
    }

    /// 使用係數進行預測
    fn predict_with_coeffs(x: &[f64], coefficients: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| {
            coefficients.iter().enumerate()
                .map(|(i, &coeff)| coeff * xi.powi(i as i32))
                .sum()
        }).collect()
    }

    /// 預測新資料點
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        Self::predict_with_coeffs(x, &self.coefficients)
    }

    /// 取得係數
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// 取得多項式階數
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// 取得決定係數 R²
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 格式化輸出模型資訊
    pub fn summary(&self) -> String {
        let mut result = format!(
            "多項式回歸模型 (階數: {})\n=======================\n決定係數 R²: {:.4}\n\n係數:\n",
            self.degree, self.r_squared
        );

        for (i, &coeff) in self.coefficients.iter().enumerate() {
            if i == 0 {
                result.push_str(&format!("常數項: {:.4}\n", coeff));
            } else if i == 1 {
                result.push_str(&format!("x: {:.4}\n", coeff));
            } else {
                result.push_str(&format!("x^{}: {:.4}\n", i, coeff));
            }
        }

        result
    }
}

/// K-最近鄰演算法 (K-Nearest Neighbors)
#[derive(Debug, Clone)]
pub struct KNNClassifier {
    k: usize,
    training_data: Vec<Vec<f64>>,
    labels: Vec<String>,
}

impl KNNClassifier {
    /// 建立 KNN 分類器
    pub fn new(k: usize) -> Self {
        Self {
            k,
            training_data: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// 訓練模型
    pub fn fit(&mut self, training_data: Vec<Vec<f64>>, labels: Vec<String>) -> Result<(), &'static str> {
        if training_data.len() != labels.len() {
            return Err("訓練資料和標籤數量不一致");
        }

        if training_data.is_empty() {
            return Err("訓練資料為空");
        }

        // 檢查所有樣本的維度是否一致
        let feature_dim = training_data[0].len();
        for sample in &training_data {
            if sample.len() != feature_dim {
                return Err("訓練樣本維度不一致");
            }
        }

        self.training_data = training_data;
        self.labels = labels;
        Ok(())
    }

    /// 預測單一樣本
    pub fn predict(&self, sample: &[f64]) -> Result<String, &'static str> {
        if self.training_data.is_empty() {
            return Err("模型尚未訓練");
        }

        if sample.len() != self.training_data[0].len() {
            return Err("樣本維度與訓練資料不一致");
        }

        // 計算與所有訓練樣本的距離
        let mut distances: Vec<(f64, usize)> = self.training_data.iter()
            .enumerate()
            .map(|(i, train_sample)| {
                let distance = Self::euclidean_distance(sample, train_sample);
                (distance, i)
            })
            .collect();

        // 排序距離
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // 找到 k 個最近鄰居的標籤
        let mut label_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for i in 0..self.k.min(distances.len()) {
            let label = &self.labels[distances[i].1];
            *label_counts.entry(label.clone()).or_insert(0) += 1;
        }

        // 返回出現最多次的標籤
        label_counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .ok_or("無法確定預測標籤")
    }

    /// 預測多個樣本
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Result<Vec<String>, &'static str> {
        samples.iter()
            .map(|sample| self.predict(sample))
            .collect()
    }

    /// 計算歐幾里德距離
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// 取得 K 值
    pub fn k(&self) -> usize {
        self.k
    }

    /// 取得訓練資料 (用於比較不同 K 值)
    pub fn training_data(&self) -> &[Vec<f64>] {
        &self.training_data
    }

    /// 取得訓練標籤 (用於比較不同 K 值)
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// 簡單的決策樹分類器 (使用 ID3 演算法的簡化版本)
#[derive(Debug, Clone)]
pub struct SimpleDecisionTree {
    root: Option<Box<TreeNode>>,
}

#[derive(Debug, Clone)]
struct TreeNode {
    feature_index: usize,
    threshold: f64,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    label: Option<String>,
}

impl SimpleDecisionTree {
    /// 建立新的決策樹
    pub fn new() -> Self {
        Self { root: None }
    }

    /// 訓練模型 (簡化版本)
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[String], max_depth: usize) -> Result<(), &'static str> {
        if features.len() != labels.len() {
            return Err("特徵和標籤數量不一致");
        }

        let indices: Vec<usize> = (0..features.len()).collect();
        self.root = Some(Box::new(self.build_tree(features, labels, &indices, 0, max_depth)));
        Ok(())
    }

    /// 遞歸建構決策樹
    fn build_tree(&self, features: &[Vec<f64>], labels: &[String], indices: &[usize], depth: usize, max_depth: usize) -> TreeNode {
        // 檢查是否為葉節點
        let unique_labels: std::collections::HashSet<&String> = indices.iter()
            .map(|&i| &labels[i])
            .collect();

        if unique_labels.len() == 1 || depth >= max_depth || indices.len() < 2 {
            let most_common = self.most_common_label(labels, indices);
            return TreeNode {
                feature_index: 0,
                threshold: 0.0,
                left: None,
                right: None,
                label: Some(most_common),
            };
        }

        // 找到最佳分割點 (簡化：使用第一個特徵的中位數)
        let feature_index = 0;
        let feature_values: Vec<f64> = indices.iter()
            .map(|&i| features[i][feature_index])
            .collect();

        let threshold = self.median(&feature_values);

        // 分割資料
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &i in indices {
            if features[i][feature_index] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        // 遞歸建構子樹
        let left = if left_indices.is_empty() {
            None
        } else {
            Some(Box::new(self.build_tree(features, labels, &left_indices, depth + 1, max_depth)))
        };

        let right = if right_indices.is_empty() {
            None
        } else {
            Some(Box::new(self.build_tree(features, labels, &right_indices, depth + 1, max_depth)))
        };

        TreeNode {
            feature_index,
            threshold,
            left,
            right,
            label: None,
        }
    }

    /// 預測單一樣本
    pub fn predict(&self, sample: &[f64]) -> Option<String> {
        self.traverse_tree(&self.root, sample)
    }

    /// 遍歷樹進行預測
    fn traverse_tree(&self, node: &Option<Box<TreeNode>>, sample: &[f64]) -> Option<String> {
        match node {
            Some(n) => {
                if let Some(ref label) = n.label {
                    return Some(label.clone());
                }

                if sample[n.feature_index] <= n.threshold {
                    self.traverse_tree(&n.left, sample)
                } else {
                    self.traverse_tree(&n.right, sample)
                }
            }
            None => None,
        }
    }

    /// 計算中位數
    fn median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }

    /// 找到最常見的標籤
    fn most_common_label(&self, labels: &[String], indices: &[usize]) -> String {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for &i in indices {
            *counts.entry(labels[i].clone()).or_insert(0) += 1;
        }

        counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .unwrap_or_else(|| "unknown".to_string())
    }
}
