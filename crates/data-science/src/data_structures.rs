//! 資料結構模組
//!
//! 提供類似 Pandas 的資料結構：Series 和 DataFrame

use std::collections::HashMap;
use crate::stats::BasicStats;

/// 類似 Pandas Series 的資料結構
#[derive(Debug, Clone)]
pub struct Series {
    name: String,
    data: Vec<f64>,
    index: Vec<String>,
}

impl Series {
    /// 建立新的 Series
    pub fn new(name: &str, data: Vec<f64>) -> Self {
        let index = (0..data.len()).map(|i| i.to_string()).collect();
        Self {
            name: name.to_string(),
            data,
            index,
        }
    }

    /// 建立帶有自訂索引的 Series
    pub fn with_index(name: &str, data: Vec<f64>, index: Vec<String>) -> Result<Self, &'static str> {
        if data.len() != index.len() {
            return Err("資料和索引長度不一致");
        }
        Ok(Self {
            name: name.to_string(),
            data,
            index,
        })
    }

    /// 取得 Series 名稱
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 取得資料
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// 取得索引
    pub fn index(&self) -> &[String] {
        &self.index
    }

    /// 取得長度
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 檢查是否為空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 根據索引取得值
    pub fn get(&self, idx: &str) -> Option<f64> {
        self.index.iter().position(|i| i == idx).map(|pos| self.data[pos])
    }

    /// 根據位置取得值
    pub fn iloc(&self, pos: usize) -> Option<f64> {
        self.data.get(pos).copied()
    }

    /// 計算統計資訊
    pub fn describe(&self) -> SeriesStats {
        let stats = BasicStats::new(&self.data);
        SeriesStats {
            count: self.len() as f64,
            mean: stats.mean(),
            std: stats.std_dev(),
            min: *self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
            max: *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0),
            median: stats.median(),
        }
    }

    /// 應用函數到每個元素
    pub fn map<F>(&self, f: F) -> Series
    where
        F: Fn(f64) -> f64,
    {
        let new_data = self.data.iter().map(|&x| f(x)).collect();
        Series::with_index(&self.name, new_data, self.index.clone()).unwrap()
    }

    /// 過濾資料
    pub fn filter<F>(&self, predicate: F) -> Series
    where
        F: Fn(f64) -> bool,
    {
        let mut new_data = Vec::new();
        let mut new_index = Vec::new();

        for (i, &value) in self.data.iter().enumerate() {
            if predicate(value) {
                new_data.push(value);
                new_index.push(self.index[i].clone());
            }
        }

        Series::with_index(&self.name, new_data, new_index).unwrap()
    }
}

/// Series 的統計摘要
#[derive(Debug, Clone)]
pub struct SeriesStats {
    pub count: f64,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

impl SeriesStats {
    /// 格式化輸出統計資訊
    pub fn format(&self) -> String {
        format!(
            "Count: {:.0}\nMean: {:.4}\nStd: {:.4}\nMin: {:.4}\nMax: {:.4}\nMedian: {:.4}",
            self.count, self.mean, self.std, self.min, self.max, self.median
        )
    }
}

/// 簡化的 DataFrame 結構
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<String>,
    data: HashMap<String, Vec<f64>>,
    index: Vec<String>,
}

impl DataFrame {
    /// 建立新的 DataFrame
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            data: HashMap::new(),
            index: Vec::new(),
        }
    }

    /// 從 HashMap 建立 DataFrame
    pub fn from_hashmap(data: HashMap<String, Vec<f64>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Ok(Self::new());
        }

        let first_col_len = data.values().next().unwrap().len();
        for (_col_name, col_data) in &data {
            if col_data.len() != first_col_len {
                return Err("欄位長度不一致");
            }
        }

        let columns = data.keys().cloned().collect();
        let index = (0..first_col_len).map(|i| i.to_string()).collect();

        Ok(Self {
            columns,
            data,
            index,
        })
    }

    /// 從 CSV-like 資料建立 DataFrame
    pub fn from_csv_data(headers: Vec<String>, rows: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        if headers.is_empty() || rows.is_empty() {
            return Err("標題或資料為空");
        }

        if headers.len() != rows[0].len() {
            return Err("標題和資料欄數不一致");
        }

        let mut data = HashMap::new();
        for (i, header) in headers.iter().enumerate() {
            let column_data: Vec<f64> = rows.iter().map(|row| row[i]).collect();
            data.insert(header.clone(), column_data);
        }

        Self::from_hashmap(data)
    }

    /// 取得欄位名稱
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// 取得資料列數
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// 檢查是否為空
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// 取得指定欄位
    pub fn get_column(&self, column: &str) -> Option<&[f64]> {
        self.data.get(column).map(|v| v.as_slice())
    }

    /// 取得指定欄位作為 Series
    pub fn get_series(&self, column: &str) -> Option<Series> {
        self.data.get(column).map(|data| {
            Series::with_index(column, data.clone(), self.index.clone()).unwrap()
        })
    }

    /// 新增欄位
    pub fn add_column(&mut self, name: String, data: Vec<f64>) -> Result<(), &'static str> {
        if data.len() != self.len() {
            return Err("新欄位長度與現有資料不一致");
        }

        if self.columns.contains(&name) {
            return Err("欄位名稱已存在");
        }

        self.columns.push(name.clone());
        self.data.insert(name, data);
        Ok(())
    }

    /// 刪除欄位
    pub fn remove_column(&mut self, name: &str) -> Result<(), &'static str> {
        if !self.columns.iter().any(|col| col == &name) {
            return Err("欄位不存在");
        }

        self.columns.retain(|c| c != name);
        self.data.remove(name);
        Ok(())
    }

    /// 取得指定列的資料
    pub fn get_row(&self, idx: usize) -> Option<HashMap<String, f64>> {
        if idx >= self.len() {
            return None;
        }

        let mut row = HashMap::new();
        for column in &self.columns {
            if let Some(col_data) = self.data.get(column) {
                row.insert(column.clone(), col_data[idx]);
            }
        }
        Some(row)
    }

    /// 計算各欄位的統計資訊
    pub fn describe(&self) -> HashMap<String, SeriesStats> {
        let mut stats = HashMap::new();
        for column in &self.columns {
            if let Some(data) = self.data.get(column) {
                let series = Series::new(column, data.clone());
                stats.insert(column.clone(), series.describe());
            }
        }
        stats
    }

    /// 計算欄位間的相關係數
    pub fn correlation_matrix(&self) -> HashMap<(String, String), f64> {
        let mut correlations = HashMap::new();

        for i in 0..self.columns.len() {
            for j in i..self.columns.len() {
                let col1 = &self.columns[i];
                let col2 = &self.columns[j];

                if let (Some(data1), Some(data2)) = (self.data.get(col1), self.data.get(col2)) {
                    if let Ok(corr) = crate::stats::correlation(data1, data2) {
                        correlations.insert((col1.clone(), col2.clone()), corr);
                        if i != j {
                            correlations.insert((col2.clone(), col1.clone()), corr);
                        }
                    }
                }
            }
        }

        correlations
    }

    /// 格式化輸出前 n 行資料
    pub fn head(&self, n: usize) -> String {
        let display_n = n.min(self.len());
        let mut output = String::new();

        // 標題列
        output.push_str("Index\t");
        for col in &self.columns {
            output.push_str(&format!("{}\t", col));
        }
        output.push('\n');

        // 分隔線
        output.push_str("-----\t");
        for _ in &self.columns {
            output.push_str("-----\t");
        }
        output.push('\n');

        // 資料列
        for i in 0..display_n {
            output.push_str(&format!("{}\t", self.index[i]));
            for col in &self.columns {
                if let Some(data) = self.data.get(col) {
                    output.push_str(&format!("{:.2}\t", data[i]));
                }
            }
            output.push('\n');
        }

        if self.len() > display_n {
            output.push_str(&format!("... (還有 {} 行)\n", self.len() - display_n));
        }

        output
    }
}
