//! 基礎統計運算模組
//!
//! 提供資料科學中最常用的統計計算函數

use std::collections::HashMap;

/// 基礎統計計算結構
#[derive(Debug, Clone)]
pub struct BasicStats {
    data: Vec<f64>,
    sorted_data: Vec<f64>,
}

impl BasicStats {
    /// 建立新的統計分析器
    pub fn new(data: &[f64]) -> Self {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            data: data.to_vec(),
            sorted_data: sorted,
        }
    }

    /// 計算平均值 (Mean)
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// 計算中位數 (Median)
    pub fn median(&self) -> f64 {
        if self.sorted_data.is_empty() {
            return 0.0;
        }

        let len = self.sorted_data.len();
        if len % 2 == 0 {
            (self.sorted_data[len / 2 - 1] + self.sorted_data[len / 2]) / 2.0
        } else {
            self.sorted_data[len / 2]
        }
    }

    /// 計算眾數 (Mode)
    pub fn mode(&self) -> Vec<f64> {
        if self.data.is_empty() {
            return vec![];
        }

        let mut frequency: HashMap<String, (f64, usize)> = HashMap::new();

        for &value in &self.data {
            let key = format!("{:.6}", value);
            frequency.entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((value, 1));
        }

        let max_freq = frequency.values().map(|(_, count)| *count).max().unwrap_or(0);
        frequency.values()
            .filter(|(_, count)| *count == max_freq)
            .map(|(value, _)| *value)
            .collect()
    }

    /// 計算變異數 (Variance)
    pub fn variance(&self) -> f64 {
        if self.data.len() <= 1 {
            return 0.0;
        }

        let mean = self.mean();
        let sum_sq_diff: f64 = self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum();

        sum_sq_diff / (self.data.len() - 1) as f64  // 樣本變異數
    }

    /// 計算標準差 (Standard Deviation)
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// 計算範圍 (Range)
    pub fn range(&self) -> f64 {
        if self.sorted_data.is_empty() {
            return 0.0;
        }
        self.sorted_data.last().unwrap() - self.sorted_data.first().unwrap()
    }

    /// 計算四分位距 (IQR)
    pub fn iqr(&self) -> f64 {
        if self.sorted_data.len() < 4 {
            return 0.0;
        }

        let q1_idx = self.sorted_data.len() / 4;
        let q3_idx = 3 * self.sorted_data.len() / 4;

        self.sorted_data[q3_idx] - self.sorted_data[q1_idx]
    }

    /// 計算偏態係數 (Skewness)
    pub fn skewness(&self) -> f64 {
        if self.data.len() <= 2 {
            return 0.0;
        }

        let mean = self.mean();
        let std_dev = self.std_dev();
        if std_dev == 0.0 {
            return 0.0;
        }

        let n = self.data.len() as f64;
        let sum_cubed: f64 = self.data.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum();

        sum_cubed * n / ((n - 1.0) * (n - 2.0))
    }

    /// 計算峰態係數 (Kurtosis)
    pub fn kurtosis(&self) -> f64 {
        if self.data.len() <= 3 {
            return 0.0;
        }

        let mean = self.mean();
        let std_dev = self.std_dev();
        if std_dev == 0.0 {
            return 0.0;
        }

        let n = self.data.len() as f64;
        let sum_fourth: f64 = self.data.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum();

        sum_fourth * n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0))
    }

    /// 取得原始資料
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// 取得排序後的資料
    pub fn sorted_data(&self) -> &[f64] {
        &self.sorted_data
    }
}

/// 相關係數計算
pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64, &'static str> {
    if x.len() != y.len() {
        return Err("資料長度不一致");
    }

    if x.len() < 2 {
        return Err("資料點數量太少");
    }

    let x_stats = BasicStats::new(x);
    let y_stats = BasicStats::new(y);

    let x_mean = x_stats.mean();
    let y_mean = y_stats.mean();

    let mut numerator = 0.0;
    let mut x_sum_sq = 0.0;
    let mut y_sum_sq = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;

        numerator += x_diff * y_diff;
        x_sum_sq += x_diff * x_diff;
        y_sum_sq += y_diff * y_diff;
    }

    let denominator = (x_sum_sq * y_sum_sq).sqrt();

    if denominator == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / denominator)
}

/// 計算百分位數 (Percentile)
pub fn percentile(data: &[f64], p: f64) -> Result<f64, &'static str> {
    if !(0.0..=100.0).contains(&p) {
        return Err("百分位數必須在 0-100 之間");
    }

    if data.is_empty() {
        return Err("資料為空");
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if p == 0.0 {
        return Ok(*sorted.first().unwrap());
    }

    if p == 100.0 {
        return Ok(*sorted.last().unwrap());
    }

    let n = sorted.len() as f64;
    let rank = p / 100.0 * (n - 1.0) + 1.0;

    let rank_floor = rank.floor() as usize;
    let rank_ceil = rank.ceil() as usize;

    if rank_floor >= sorted.len() {
        return Ok(*sorted.last().unwrap());
    }

    if rank_ceil >= sorted.len() {
        return Ok(sorted[rank_floor - 1]);
    }

    if rank_floor == rank_ceil {
        return Ok(sorted[rank_floor - 1]);
    }

    let weight = rank - rank_floor as f64;
    let value_floor = sorted[rank_floor - 1];
    let value_ceil = sorted[rank_ceil - 1];

    Ok(value_floor + weight * (value_ceil - value_floor))
}
