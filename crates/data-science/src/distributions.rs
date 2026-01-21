//! 機率分布模組
//!
//! 提供常見的機率分布計算和應用

use std::f64::consts::PI;

/// 正態分布 (Normal Distribution)
#[derive(Debug, Clone)]
pub struct NormalDistribution {
    mean: f64,
    std_dev: f64,
}

impl NormalDistribution {
    /// 建立正態分布
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, &'static str> {
        if std_dev <= 0.0 {
            return Err("標準差必須大於 0");
        }
        Ok(Self { mean, std_dev })
    }

    /// 標準正態分布
    pub fn standard() -> Self {
        Self { mean: 0.0, std_dev: 1.0 }
    }

    /// 計算機率密度函數 (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        let coefficient = 1.0 / (self.std_dev * (2.0 * PI).sqrt());
        let exponent = -0.5 * ((x - self.mean) / self.std_dev).powi(2);
        coefficient * exponent.exp()
    }

    /// 計算累積分布函數 (CDF) - 使用近似算法
    pub fn cdf(&self, x: f64) -> f64 {
        // 標準化
        let z = (x - self.mean) / self.std_dev;
        Self::standard_normal_cdf(z)
    }

    /// 標準正態分布的 CDF (使用近似公式)
    fn standard_normal_cdf(z: f64) -> f64 {
        // Abramowitz & Stegun approximation
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;

        let sign = if z < 0.0 { -1.0 } else { 1.0 };
        let z = z.abs();

        let t = 1.0 / (1.0 + p * z);
        let erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z).exp();

        0.5 * (1.0 + sign * erf)
    }

    /// 計算分位數 (Quantile) - 反函數
    pub fn quantile(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            panic!("機率值必須在 0 到 1 之間");
        }

        // 使用標準正態分布的分位數再轉換
        let z = Self::standard_normal_quantile(p);
        self.mean + z * self.std_dev
    }

    /// 標準正態分布的分位數
    fn standard_normal_quantile(p: f64) -> f64 {
        // 使用近似公式
        if p == 0.5 {
            return 0.0;
        }

        let q = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * (2.0 * q).ln()).sqrt();

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d0 = 1.432788;
        let d1 = 0.189269;
        let d2 = 0.001308;

        let numerator = c0 + c1 * t + c2 * t * t;
        let denominator = 1.0 + d0 * t + d1 * t * t + d2 * t * t * t;

        let z = t - numerator / denominator;

        if p < 0.5 { -z } else { z }
    }

    /// 從資料估計參數
    pub fn from_data(data: &[f64]) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("資料為空");
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;

        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);

        let std_dev = variance.sqrt();

        Self::new(mean, std_dev)
    }

    /// 取得平均數
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// 取得標準差
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

/// 學生 t 分布 (Student's t-Distribution)
#[derive(Debug, Clone)]
pub struct TDistribution {
    degrees_of_freedom: f64,
}

impl TDistribution {
    /// 建立 t 分布
    pub fn new(degrees_of_freedom: f64) -> Result<Self, &'static str> {
        if degrees_of_freedom <= 0.0 {
            return Err("自由度必須大於 0");
        }
        Ok(Self { degrees_of_freedom })
    }

    /// 計算機率密度函數 (PDF)
    pub fn pdf(&self, t: f64) -> f64 {
        let df = self.degrees_of_freedom;
        let numerator = Self::gamma((df + 1.0) / 2.0);
        let denominator = (df * PI).sqrt() * Self::gamma(df / 2.0);
        let coefficient = numerator / denominator;

        let power = -((df + 1.0) / 2.0);
        coefficient * (1.0 + (t * t) / df).powf(power)
    }

    /// 計算累積分布函數 (CDF) - 簡化近似
    pub fn cdf(&self, t: f64) -> f64 {
        // 使用近似公式，實際應用中可能需要更精確的方法
        if self.degrees_of_freedom > 30.0 {
            // 當自由度夠大時，接近正態分布
            NormalDistribution::standard().cdf(t)
        } else {
            // 簡單的近似
            0.5 + 0.5 * Self::error_function(t / (1.0 + self.degrees_of_freedom / 4.0).sqrt())
        }
    }

    /// Gamma 函數近似
    fn gamma(z: f64) -> f64 {
        // Lanczos approximation
        let g = 7.0;
        let p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                 771.32342877765313, -176.61502916214059, 12.507343278686905,
                 -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

        if z < 0.5 {
            PI / ((PI * z).sin() * Self::gamma(1.0 - z))
        } else {
            let z = z - 1.0;
            let mut x = p[0];
            for (i, &pi) in p.iter().enumerate().skip(1) {
                x += pi / (z + i as f64);
            }
            let t = z + g + 0.5;
            (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
        }
    }

    /// 誤差函數近似
    fn error_function(x: f64) -> f64 {
        // Abramowitz & Stegun approximation
        let a1 =  0.886226899;
        let a2 = -1.645349621;
        let a3 =  0.914624893;
        let a4 = -0.140543331;
        let p  =  0.886226899;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let erf = 1.0 - (((((a4 * t + a3) * t) + a2) * t + a1) * t) * (-x * x).exp();

        sign * erf
    }

    /// 取得自由度
    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }
}

/// 二項分布 (Binomial Distribution)
#[derive(Debug, Clone)]
pub struct BinomialDistribution {
    n: usize,  // 試驗次數
    p: f64,    // 成功機率
}

impl BinomialDistribution {
    /// 建立二項分布
    pub fn new(n: usize, p: f64) -> Result<Self, &'static str> {
        if !(0.0..=1.0).contains(&p) {
            return Err("成功機率必須在 0 到 1 之間");
        }
        Ok(Self { n, p })
    }

    /// 計算機率質量函數 (PMF)
    pub fn pmf(&self, k: usize) -> f64 {
        if k > self.n {
            return 0.0;
        }

        let binomial_coeff = Self::binomial_coefficient(self.n, k);
        binomial_coeff * self.p.powi(k as i32) * (1.0 - self.p).powi((self.n - k) as i32)
    }

    /// 計算累積分布函數 (CDF)
    pub fn cdf(&self, k: usize) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }

    /// 計算期望值
    pub fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }

    /// 計算變異數
    pub fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }

    /// 計算二項式係數 C(n, k)
    fn binomial_coefficient(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        if k == 0 || k == n {
            return 1.0;
        }

        // 使用循環避免大數溢出
        let k = k.min(n - k);
        let mut result = 1.0;

        for i in 1..=k {
            result *= (n - k + i) as f64 / i as f64;
        }

        result
    }

    /// 取得試驗次數
    pub fn n(&self) -> usize {
        self.n
    }

    /// 取得成功機率
    pub fn p(&self) -> f64 {
        self.p
    }
}

/// 泊松分布 (Poisson Distribution)
#[derive(Debug, Clone)]
pub struct PoissonDistribution {
    lambda: f64,  // 平均發生次數
}

impl PoissonDistribution {
    /// 建立泊松分布
    pub fn new(lambda: f64) -> Result<Self, &'static str> {
        if lambda <= 0.0 {
            return Err("平均發生次數必須大於 0");
        }
        Ok(Self { lambda })
    }

    /// 計算機率質量函數 (PMF)
    pub fn pmf(&self, k: usize) -> f64 {
        let k_factorial = Self::factorial(k);
        (self.lambda.powi(k as i32) * (-self.lambda).exp()) / k_factorial
    }

    /// 計算累積分布函數 (CDF)
    pub fn cdf(&self, k: usize) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }

    /// 計算期望值 (等於 lambda)
    pub fn mean(&self) -> f64 {
        self.lambda
    }

    /// 計算變異數 (等於 lambda)
    pub fn variance(&self) -> f64 {
        self.lambda
    }

    /// 計算階乘
    fn factorial(n: usize) -> f64 {
        if n == 0 || n == 1 {
            1.0
        } else {
            (1..=n).map(|i| i as f64).product()
        }
    }

    /// 取得 lambda 參數
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// 卡方分布 (Chi-Squared Distribution)
#[derive(Debug, Clone)]
pub struct ChiSquaredDistribution {
    degrees_of_freedom: usize,
}

impl ChiSquaredDistribution {
    /// 建立卡方分布
    pub fn new(degrees_of_freedom: usize) -> Result<Self, &'static str> {
        if degrees_of_freedom == 0 {
            return Err("自由度必須大於 0");
        }
        Ok(Self { degrees_of_freedom })
    }

    /// 計算機率密度函數 (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        let k = self.degrees_of_freedom as f64;
        let numerator = x.powf((k / 2.0) - 1.0) * (-x / 2.0).exp();
        let denominator = (2.0_f64).powf(k / 2.0) * TDistribution::gamma(k / 2.0);

        numerator / denominator
    }

    /// 計算期望值
    pub fn mean(&self) -> f64 {
        self.degrees_of_freedom as f64
    }

    /// 計算變異數
    pub fn variance(&self) -> f64 {
        2.0 * self.degrees_of_freedom as f64
    }

    /// 取得自由度
    pub fn degrees_of_freedom(&self) -> usize {
        self.degrees_of_freedom
    }
}

/// 指數分布 (Exponential Distribution)
#[derive(Debug, Clone)]
pub struct ExponentialDistribution {
    lambda: f64,  // 速率參數
}

impl ExponentialDistribution {
    /// 建立指數分布
    pub fn new(lambda: f64) -> Result<Self, &'static str> {
        if lambda <= 0.0 {
            return Err("速率參數必須大於 0");
        }
        Ok(Self { lambda })
    }

    /// 建立以平均值參數化的指數分布
    pub fn from_mean(mean: f64) -> Result<Self, &'static str> {
        if mean <= 0.0 {
            return Err("平均值必須大於 0");
        }
        Self::new(1.0 / mean)
    }

    /// 計算機率密度函數 (PDF)
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.lambda * (-self.lambda * x).exp()
        }
    }

    /// 計算累積分布函數 (CDF)
    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.lambda * x).exp()
        }
    }

    /// 計算生存函數 (Survival Function)
    pub fn survival(&self, x: f64) -> f64 {
        if x < 0.0 {
            1.0
        } else {
            (-self.lambda * x).exp()
        }
    }

    /// 計算期望值
    pub fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    /// 計算變異數
    pub fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }

    /// 取得速率參數
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// 測試函數：生成隨機樣本 (使用簡單的線性同餘產生器)
pub fn generate_random_sample(distribution: &dyn Distribution, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SimpleRng::new(seed);
    (0..n).map(|_| distribution.sample(&mut rng)).collect()
}

/// 分布特徵
pub trait Distribution {
    fn sample(&self, rng: &mut SimpleRng) -> f64;
}

/// 簡單的隨機數產生器
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        // 線性同餘產生器
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.state % (1 << 31)) as f64 / (1 << 31) as f64
    }
}

// 實作 Distribution 特徵給各個分布
impl Distribution for NormalDistribution {
    fn sample(&self, rng: &mut SimpleRng) -> f64 {
        // Box-Muller 變換
        let u1 = rng.next_f64();
        let u2 = rng.next_f64();

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        self.mean + self.std_dev * z
    }
}

impl Distribution for ExponentialDistribution {
    fn sample(&self, rng: &mut SimpleRng) -> f64 {
        let u = rng.next_f64();
        -u.ln() / self.lambda
    }
}
