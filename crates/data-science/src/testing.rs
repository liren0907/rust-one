//! 假設檢定模組
//!
//! 提供常見的統計假設檢定方法

use crate::stats::BasicStats;

/// 檢定結果結構
#[derive(Debug, Clone)]
pub struct TestResult {
    pub statistic: f64,           // 檢定統計量
    pub p_value: f64,             // p 值
    pub degrees_of_freedom: f64,  // 自由度
    pub reject_null: bool,        // 是否拒絕虛無假設
    pub alpha: f64,              // 顯著性水準
    pub test_name: String,        // 檢定名稱
}

impl TestResult {
    /// 格式化輸出檢定結果
    pub fn summary(&self) -> String {
        format!(
            "{} 檢定結果\n================\n檢定統計量: {:.4}\np 值: {:.4}\n自由度: {:.1}\n顯著性水準: {:.3}\n{} 虛無假設 (α = {:.3})",
            self.test_name,
            self.statistic,
            self.p_value,
            self.degrees_of_freedom,
            self.alpha,
            if self.reject_null { "拒絕" } else { "不拒絕" },
            self.alpha
        )
    }
}

/// 單一樣本 t 檢定 (One-sample t-test)
pub fn one_sample_t_test(sample: &[f64], mu: f64, alpha: f64) -> Result<TestResult, &'static str> {
    if sample.len() < 2 {
        return Err("樣本數量太少");
    }

    let stats = BasicStats::new(sample);
    let n = sample.len() as f64;
    let df = n - 1.0;

    // t 統計量
    let t_statistic = (stats.mean() - mu) / (stats.std_dev() / n.sqrt());

    // 計算 p 值 (雙尾檢定)
    let p_value = 2.0 * (1.0 - t_cdf(t_statistic.abs(), df));

    let reject_null = p_value < alpha;

    Ok(TestResult {
        statistic: t_statistic,
        p_value,
        degrees_of_freedom: df,
        reject_null,
        alpha,
        test_name: format!("單一樣本 t 檢定 (H₀: μ = {:.2})", mu),
    })
}

/// 獨立樣本 t 檢定 (Independent samples t-test)
pub fn independent_t_test(sample1: &[f64], sample2: &[f64], alpha: f64) -> Result<TestResult, &'static str> {
    if sample1.len() < 2 || sample2.len() < 2 {
        return Err("任一樣本數量太少");
    }

    let stats1 = BasicStats::new(sample1);
    let stats2 = BasicStats::new(sample2);

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    let df = n1 + n2 - 2.0;

    // 合併變異數
    let pooled_variance = ((n1 - 1.0) * stats1.variance() + (n2 - 1.0) * stats2.variance()) / df;
    let se = (pooled_variance * (1.0/n1 + 1.0/n2)).sqrt();

    if se == 0.0 {
        return Err("標準誤為零，無法進行檢定");
    }

    // t 統計量
    let t_statistic = (stats1.mean() - stats2.mean()) / se;

    // 計算 p 值 (雙尾檢定)
    let p_value = 2.0 * (1.0 - t_cdf(t_statistic.abs(), df));

    let reject_null = p_value < alpha;

    Ok(TestResult {
        statistic: t_statistic,
        p_value,
        degrees_of_freedom: df,
        reject_null,
        alpha,
        test_name: "獨立樣本 t 檢定 (H₀: μ₁ = μ₂)".to_string(),
    })
}

/// 配對樣本 t 檢定 (Paired t-test)
pub fn paired_t_test(before: &[f64], after: &[f64], alpha: f64) -> Result<TestResult, &'static str> {
    if before.len() != after.len() {
        return Err("配對樣本長度不一致");
    }

    if before.len() < 2 {
        return Err("樣本數量太少");
    }

    // 計算差值
    let differences: Vec<f64> = before.iter().zip(after.iter())
        .map(|(&b, &a)| a - b)
        .collect();

    let diff_stats = BasicStats::new(&differences);
    let n = differences.len() as f64;
    let df = n - 1.0;

    // t 統計量
    let t_statistic = diff_stats.mean() / (diff_stats.std_dev() / n.sqrt());

    // 計算 p 值 (雙尾檢定)
    let p_value = 2.0 * (1.0 - t_cdf(t_statistic.abs(), df));

    let reject_null = p_value < alpha;

    Ok(TestResult {
        statistic: t_statistic,
        p_value,
        degrees_of_freedom: df,
        reject_null,
        alpha,
        test_name: "配對樣本 t 檢定 (H₀: 差值平均 = 0)".to_string(),
    })
}

/// 單一樣本比例檢定 (One-sample proportion test)
pub fn one_sample_proportion_test(successes: usize, n: usize, p0: f64, alpha: f64) -> Result<TestResult, &'static str> {
    if n == 0 {
        return Err("樣本數量不能為零");
    }

    if successes > n {
        return Err("成功次數不能大於總次數");
    }

    if !(0.0..=1.0).contains(&p0) {
        return Err("虛無假設的比例必須在 0 到 1 之間");
    }

    let p_hat = successes as f64 / n as f64;

    // z 統計量
    let se = (p0 * (1.0 - p0) / n as f64).sqrt();
    if se == 0.0 {
        return Err("標準誤為零，無法進行檢定");
    }

    let z_statistic = (p_hat - p0) / se;

    // 雙尾檢定的 p 值
    let p_value = 2.0 * (1.0 - normal_cdf(z_statistic.abs()));

    let reject_null = p_value < alpha;

    Ok(TestResult {
        statistic: z_statistic,
        p_value,
        degrees_of_freedom: (n - 1) as f64,  // 近似
        reject_null,
        alpha,
        test_name: format!("單一樣本比例檢定 (H₀: p = {:.3})", p0),
    })
}

/// 卡方適配度檢定 (Chi-square goodness of fit test)
pub fn chi_square_goodness_of_fit(observed: &[usize], expected: &[f64], alpha: f64) -> Result<TestResult, &'static str> {
    if observed.len() != expected.len() {
        return Err("觀察值和期望值長度不一致");
    }

    if observed.len() < 2 {
        return Err("類別數量太少");
    }

    // 計算卡方統計量
    let mut chi_square = 0.0;
    let mut df = 0.0;

    for (_i, (&obs, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
        if exp > 0.0 {
            chi_square += ((obs as f64 - exp).powi(2)) / exp;
            df += 1.0;
        }
    }

    df -= 1.0;  // 自由度 = 類別數 - 1

    if df <= 0.0 {
        return Err("自由度無效");
    }

    // 計算 p 值
    let p_value = 1.0 - chi_square_cdf(chi_square, df as usize);

    let reject_null = p_value < alpha;

    Ok(TestResult {
        statistic: chi_square,
        p_value,
        degrees_of_freedom: df,
        reject_null,
        alpha,
        test_name: "卡方適配度檢定".to_string(),
    })
}

/// 卡方獨立性檢定 (Chi-square test of independence)
pub fn chi_square_independence(contingency_table: &[Vec<usize>], alpha: f64) -> Result<TestResult, &'static str> {
    if contingency_table.is_empty() || contingency_table[0].is_empty() {
        return Err("列聯表為空");
    }

    let rows = contingency_table.len();
    let cols = contingency_table[0].len();

    // 檢查所有行長度一致
    for row in contingency_table {
        if row.len() != cols {
            return Err("列聯表行長度不一致");
        }
    }

    // 計算行和列總和
    let mut row_totals = vec![0; rows];
    let mut col_totals = vec![0; cols];
    let mut grand_total = 0;

    for i in 0..rows {
        for j in 0..cols {
            row_totals[i] += contingency_table[i][j];
            col_totals[j] += contingency_table[i][j];
            grand_total += contingency_table[i][j];
        }
    }

    if grand_total == 0 {
        return Err("總觀察數為零");
    }

    // 計算卡方統計量
    let mut chi_square = 0.0;

    for i in 0..rows {
        for j in 0..cols {
            let expected = (row_totals[i] as f64 * col_totals[j] as f64) / grand_total as f64;
            if expected > 0.0 {
                let observed = contingency_table[i][j] as f64;
                chi_square += ((observed - expected).powi(2)) / expected;
            }
        }
    }

    // 自由度 = (行數 - 1) * (列數 - 1)
    let df = ((rows - 1) * (cols - 1)) as f64;

    // 計算 p 值
    let p_value = 1.0 - chi_square_cdf(chi_square, df as usize);

    let reject_null = p_value < alpha;

    Ok(TestResult {
        statistic: chi_square,
        p_value,
        degrees_of_freedom: df,
        reject_null,
        alpha,
        test_name: "卡方獨立性檢定".to_string(),
    })
}

/// F 檢定 (F-test for comparing variances)
pub fn f_test(sample1: &[f64], sample2: &[f64], alpha: f64) -> Result<TestResult, &'static str> {
    if sample1.len() < 2 || sample2.len() < 2 {
        return Err("任一樣本數量太少");
    }

    let stats1 = BasicStats::new(sample1);
    let stats2 = BasicStats::new(sample2);

    let var1 = stats1.variance();
    let var2 = stats2.variance();

    if var1 == 0.0 || var2 == 0.0 {
        return Err("樣本變異數為零");
    }

    // F 統計量 (較大的變異數在分子)
    let f_statistic = if var1 >= var2 {
        var1 / var2
    } else {
        var2 / var1
    };

    let df1 = sample1.len() as f64 - 1.0;
    let df2 = sample2.len() as f64 - 1.0;

    // 計算 p 值 (雙尾檢定)
    let p_value = 2.0 * (1.0 - f_cdf(f_statistic, df1 as usize, df2 as usize));

    let reject_null = p_value < alpha;

    Ok(TestResult {
        statistic: f_statistic,
        p_value,
        degrees_of_freedom: df1.min(df2),  // 簡化處理
        reject_null,
        alpha,
        test_name: "F 檢定 (H₀: σ₁² = σ₂²)".to_string(),
    })
}

/// 輔助函數：t 分布的累積分布函數近似
fn t_cdf(t: f64, df: f64) -> f64 {
    // 使用正態近似當自由度夠大
    if df > 30.0 {
        normal_cdf(t)
    } else {
        // 簡化的 t 分布 CDF 近似
        let z = t * (1.0 - 1.0/(4.0 * df)).sqrt();
        normal_cdf(z)
    }
}

/// 輔助函數：標準正態分布的累積分布函數
fn normal_cdf(z: f64) -> f64 {
    // 使用近似公式
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let z = z.abs();

    let t = 1.0 / (1.0 + p * z);
    let erf = 1.0 - ((((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (-z * z).exp());

    0.5 * (1.0 + sign * erf)
}

/// 輔助函數：卡方分布的累積分布函數近似
fn chi_square_cdf(chi2: f64, df: usize) -> f64 {
    // 使用正態近似當自由度夠大
    if df > 30 {
        let mean = df as f64;
        let std = (2.0 * df as f64).sqrt();
        let z = (chi2 - mean) / std;
        normal_cdf(z)
    } else {
        // 簡化的卡方分布 CDF 近似
        let k = df as f64;
        if chi2 < k {
            // 使用 Gamma 分布近似
            gamma_cdf(chi2 / 2.0, k / 2.0)
        } else {
            1.0 - gamma_cdf(chi2 / 2.0, k / 2.0)
        }
    }
}

/// 輔助函數：Gamma 分布的累積分布函數近似
fn gamma_cdf(x: f64, shape: f64) -> f64 {
    // 簡化的近似
    if x <= 0.0 {
        0.0
    } else if shape >= 1.0 {
        // 使用 Wilson-Hilferty 變換近似正態分布
        let mean = shape;
        let variance = shape;
        let z = (x - mean) / variance.sqrt();
        normal_cdf(z)
    } else {
        // 簡單的近似
        (x / (x + shape)).powf(shape)
    }
}

/// 輔助函數：F 分布的累積分布函數近似
fn f_cdf(f: f64, df1: usize, df2: usize) -> f64 {
    // 簡化的 F 分布 CDF 近似
    let d1 = df1 as f64;
    let d2 = df2 as f64;

    if f <= 1.0 {
        // 使用 Beta 分布近似
        beta_cdf(f * d1 / (f * d1 + d2), d1 / 2.0, d2 / 2.0)
    } else {
        1.0 - beta_cdf(d1 / (f * d1 + d2), d2 / 2.0, d1 / 2.0)
    }
}

/// 輔助函數：Beta 分布的累積分布函數近似
fn beta_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    // 使用正規近似當參數夠大
    if alpha > 5.0 && beta > 5.0 {
        let mean = alpha / (alpha + beta);
        let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
        let z = (x - mean) / variance.sqrt();
        normal_cdf(z)
    } else {
        // 簡單的近似
        let incomplete_beta = x.powf(alpha) * (1.0 - x).powf(beta);
        incomplete_beta / (incomplete_beta + 1.0)
    }
}
