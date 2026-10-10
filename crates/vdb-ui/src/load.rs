//! Read embeddings from a local file, a folder, or an https URL, then build
//! an in-memory index. The UI calls this and forwards each event to the page.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use vectordb::{Engine, EngineConfig, HnswParams, Metric};

const URL_CAP: u64 = 512 * 1024 * 1024;
const LOG_EVERY: usize = 1_000;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub location: String,
    pub limit: usize,
    pub metric: Metric,
    pub index: String,
}

#[derive(Debug, Clone)]
pub enum Event {
    Stage(&'static str),
    Log(String),
    Progress { done: u64, total: u64 },
}

#[derive(Debug, Clone)]
pub struct Label {
    pub id: u64,
    pub title: String,
    pub text: String,
}

pub struct Loaded {
    pub engine: Engine,
    pub labels: Vec<Label>,
    /// Copy of each inserted vector so the page can search by row id.
    pub vectors: Vec<Vec<f32>>,
    pub dim: usize,
    pub build_ms: u64,
    pub metric: String,
    pub index: String,
}

struct Row {
    source_id: String,
    title: String,
    text: String,
    vector: Vec<f32>,
}

pub fn load_dataset(
    options: &LoadOptions,
    mut on_event: impl FnMut(Event),
) -> Result<Loaded, String> {
    if options.limit == 0 {
        return Err("limit must be at least 1".into());
    }
    on_event(Event::Stage("resolve"));
    let location = options.location.trim();
    if location.is_empty() {
        return Err("location is empty".into());
    }

    let rows = if location == "sample://clusters" {
        on_event(Event::Log(
            "using the built-in sample: 24 vectors in three clusters".into(),
        ));
        let mut rows = sample_rows();
        rows.truncate(options.limit);
        rows
    } else if location.starts_with("http://") || location.starts_with("https://") {
        let file = download(location, &mut on_event)?;
        let found = read_files(
            std::slice::from_ref(&file.path),
            options.limit,
            &mut on_event,
        )?;
        drop(file);
        found
    } else {
        let path = resolve_user_path(location);
        if !path.exists() {
            return Err(format!("path does not exist: {}", path.display()));
        }
        on_event(Event::Log(format!("resolved {}", path.display())));
        let files = if path.is_dir() {
            let mut found = Vec::new();
            collect_files(&path, 0, &mut found)?;
            found.sort();
            if found.is_empty() {
                return Err(format!(
                    "no .parquet, .csv, or .jsonl files under {}",
                    path.display()
                ));
            }
            on_event(Event::Log(format!(
                "found {} data file{}",
                found.len(),
                if found.len() == 1 { "" } else { "s" }
            )));
            found
        } else {
            vec![path]
        };
        read_files(&files, options.limit, &mut on_event)?
    };

    if rows.is_empty() {
        return Err("the dataset has no embedding rows".into());
    }
    let dim = rows[0].vector.len();
    on_event(Event::Stage("index"));
    on_event(Event::Log(format!(
        "building {} index, {} × {}-d, metric {}",
        options.index,
        rows.len(),
        dim,
        options.metric.name()
    )));
    let started = Instant::now();
    let mut config = match options.index.as_str() {
        "flat" => EngineConfig::flat(dim, options.metric),
        "hnsw" => EngineConfig::hnsw(dim, options.metric, HnswParams::default()),
        other => return Err(format!("unknown index `{other}` (use hnsw or flat)")),
    };
    config.sync = vectordb::SyncPolicy::Never;
    let mut engine = Engine::open_memory(config).map_err(|err| err.to_string())?;
    let total = rows.len() as u64;
    let mut labels = Vec::with_capacity(rows.len());
    let mut vectors = Vec::with_capacity(rows.len());
    for (i, row) in rows.into_iter().enumerate() {
        let id = i as u64;
        engine
            .upsert(id, &row.vector)
            .map_err(|err| err.to_string())?;
        labels.push(Label {
            id,
            title: display_title(&row, id),
            text: clip(&row.text, 280),
        });
        vectors.push(row.vector);
        if (i + 1) % LOG_EVERY == 0 || (i + 1) as u64 == total {
            on_event(Event::Progress {
                done: (i + 1) as u64,
                total,
            });
        }
    }
    let build_ms = started.elapsed().as_millis() as u64;
    on_event(Event::Log(format!(
        "ready: {} vectors, build {build_ms} ms",
        labels.len()
    )));
    on_event(Event::Stage("ready"));
    Ok(Loaded {
        engine,
        labels,
        vectors,
        dim,
        build_ms,
        metric: options.metric.name().to_string(),
        index: options.index.clone(),
    })
}

fn display_title(row: &Row, id: u64) -> String {
    if !row.title.is_empty() {
        clip(&row.title, 160)
    } else if !row.source_id.is_empty() {
        clip(&row.source_id, 160)
    } else {
        format!("row {id}")
    }
}

fn clip(text: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, ch) in text.chars().enumerate() {
        if i >= max {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn sample_rows() -> Vec<Row> {
    let groups = [
        ("red", [1.0_f32, 0.0, 0.0, 0.0]),
        ("green", [0.0, 1.0, 0.0, 0.0]),
        ("blue", [0.0, 0.0, 1.0, 0.0]),
    ];
    let mut rows = Vec::new();
    for (name, center) in groups {
        for i in 0..8 {
            let mut vector = center;
            vector[3] = i as f32 * 0.03;
            rows.push(Row {
                source_id: format!("{name}-{i}"),
                title: format!("{name} {i}"),
                text: format!("sample point in the {name} group"),
                vector: vector.to_vec(),
            });
        }
    }
    rows
}

struct TempDownload {
    path: PathBuf,
}

impl Drop for TempDownload {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn download(url: &str, on_event: &mut impl FnMut(Event)) -> Result<TempDownload, String> {
    on_event(Event::Stage("download"));
    on_event(Event::Log(format!("downloading {url}")));
    let response = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|err| err.to_string())?
        .get(url)
        .send()
        .map_err(|err| err.to_string())?;
    if !response.status().is_success() {
        return Err(format!("download failed: HTTP {}", response.status()));
    }
    if let Some(len) = response.content_length() {
        if len > URL_CAP {
            return Err(format!(
                "this URL is {} MB. Download it locally and paste the folder path. URL loads stop at 512 MB.",
                len / (1024 * 1024)
            ));
        }
    }
    let path = std::env::temp_dir().join(format!("vdb-ui-download-{}", std::process::id()));
    let mut file = File::create(&path).map_err(|err| err.to_string())?;
    let mut reader = response;
    let mut buf = [0u8; 64 * 1024];
    let mut got = 0u64;
    let mut next_log = 1024 * 1024;
    loop {
        let n = reader.read(&mut buf).map_err(|err| err.to_string())?;
        if n == 0 {
            break;
        }
        got += n as u64;
        if got > URL_CAP {
            let _ = fs::remove_file(&path);
            return Err(
                "download passed 512 MB. Download the dataset locally and paste the folder path."
                    .into(),
            );
        }
        file.write_all(&buf[..n]).map_err(|err| err.to_string())?;
        if got >= next_log {
            on_event(Event::Log(format!("downloaded {} MB", got / (1024 * 1024))));
            next_log += 8 * 1024 * 1024;
        }
    }
    on_event(Event::Log(format!(
        "saved download ({} MB) for parsing",
        got / (1024 * 1024)
    )));
    Ok(TempDownload { path })
}

fn collect_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if depth > 3 || out.len() > 2_000 {
        return Ok(());
    }
    let entries = fs::read_dir(dir).map_err(|err| format!("{}: {err}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, depth + 1, out)?;
        } else if is_table(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_table(path: &Path) -> bool {
    matches!(
        extension(path).as_deref(),
        Some("parquet" | "csv" | "tsv" | "jsonl" | "ndjson")
    )
}

/// Expand `~` and a doubled home prefix.
///
/// `~/Downloads/a.tsv` is the home directory plus `Downloads/a.tsv`.
/// `~/Users/<name>/Downloads/a.tsv` is the same file with the home folder
/// written twice, which is what a location field produces when `~` is
/// pasted in front of an absolute path.
fn resolve_user_path(location: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    resolve_with_home(location, home.as_deref())
}

fn resolve_with_home(location: &str, home: Option<&Path>) -> PathBuf {
    let Some(rest) = location.strip_prefix("~/").or_else(|| {
        if location == "~" {
            Some("")
        } else {
            None
        }
    }) else {
        return PathBuf::from(location);
    };
    let Some(home) = home else {
        return PathBuf::from(location);
    };
    if rest.is_empty() {
        return home.to_path_buf();
    }
    let direct = home.join(rest);
    if direct.exists() {
        return direct;
    }
    let Some(user) = home.file_name().and_then(|name| name.to_str()) else {
        return direct;
    };
    let doubled = format!("Users/{user}/");
    if let Some(after) = rest.strip_prefix(&doubled) {
        let corrected = home.join(after);
        if corrected.exists() {
            return corrected;
        }
    }
    direct
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn read_files(
    files: &[PathBuf],
    limit: usize,
    on_event: &mut impl FnMut(Event),
) -> Result<Vec<Row>, String> {
    on_event(Event::Stage("parse"));
    let mut rows = Vec::new();
    for path in files {
        if rows.len() >= limit {
            break;
        }
        let kind = sniff(path)?;
        on_event(Event::Log(format!("reading {}", path.display())));
        match kind {
            Kind::Parquet => read_parquet(path, limit, &mut rows, on_event)?,
            Kind::Csv => read_csv(path, b',', limit, &mut rows, on_event)?,
            Kind::Tsv => read_csv(path, b'\t', limit, &mut rows, on_event)?,
            Kind::Jsonl => read_jsonl(path, limit, &mut rows, on_event)?,
        }
        on_event(Event::Progress {
            done: rows.len() as u64,
            total: limit as u64,
        });
    }
    Ok(rows)
}

enum Kind {
    Parquet,
    Csv,
    Tsv,
    Jsonl,
}

fn sniff(path: &Path) -> Result<Kind, String> {
    match extension(path).as_deref() {
        Some("parquet") => return Ok(Kind::Parquet),
        Some("csv") => return Ok(Kind::Csv),
        Some("tsv") => return Ok(Kind::Tsv),
        Some("jsonl") | Some("ndjson") => return Ok(Kind::Jsonl),
        _ => {}
    }
    let mut file = File::open(path).map_err(|err| err.to_string())?;
    let mut magic = [0u8; 4];
    let n = file.read(&mut magic).map_err(|err| err.to_string())?;
    if n >= 4 && &magic == b"PAR1" {
        return Ok(Kind::Parquet);
    }
    Err(format!(
        "{} is not parquet, csv, tsv, or jsonl",
        path.display()
    ))
}

fn read_parquet(
    path: &Path,
    limit: usize,
    rows: &mut Vec<Row>,
    on_event: &mut impl FnMut(Event),
) -> Result<(), String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|err| err.to_string())?;
    let arrow_schema = builder.schema().clone();
    let names: Vec<String> = arrow_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let emb_name = find_name(&names, &["emb", "embedding", "embeddings", "vector"])
        .ok_or_else(|| format!("no embedding column in {names:?}"))?;
    let title_name = find_name(&names, &["title", "name"]);
    let text_name = find_name(&names, &["text", "content", "overview", "body", "abstract"]);
    let id_name = find_name(&names, &["id", "wiki_id"]);
    on_event(Event::Log(format!(
        "columns: embedding `{emb_name}`{}{}",
        title_name
            .as_ref()
            .map(|name| format!(", title `{name}`"))
            .unwrap_or_default(),
        text_name
            .as_ref()
            .map(|name| format!(", text `{name}`"))
            .unwrap_or_default()
    )));
    let file_rows = builder.metadata().file_metadata().num_rows() as u64;
    on_event(Event::Log(format!(
        "{} rows in this file, keeping at most {limit}",
        file_rows
    )));

    let mut root_indexes = vec![name_index(&names, &emb_name)];
    for extra in [&title_name, &text_name, &id_name].into_iter().flatten() {
        root_indexes.push(name_index(&names, extra));
    }
    let mask = ProjectionMask::roots(
        builder.metadata().file_metadata().schema_descr(),
        root_indexes,
    );
    let reader = builder
        .with_projection(mask)
        .with_batch_size(1024)
        .build()
        .map_err(|err| err.to_string())?;

    let mut skipped = 0usize;
    for batch in reader {
        let batch = batch.map_err(|err| err.to_string())?;
        append_batch(
            &batch,
            &Columns {
                emb: &emb_name,
                title: title_name.as_deref(),
                text: text_name.as_deref(),
                id: id_name.as_deref(),
            },
            limit,
            rows,
            &mut skipped,
        )?;
        if rows.len() % LOG_EVERY < 1024 {
            on_event(Event::Progress {
                done: rows.len() as u64,
                total: limit as u64,
            });
        }
        if rows.len() >= limit {
            break;
        }
    }
    if skipped > 0 {
        on_event(Event::Log(format!("skipped {skipped} empty embeddings")));
    }
    Ok(())
}

fn name_index(names: &[String], name: &str) -> usize {
    names.iter().position(|item| item == name).expect("column")
}

fn find_name(names: &[String], wanted: &[&str]) -> Option<String> {
    names
        .iter()
        .find(|name| {
            let lower = name.to_ascii_lowercase();
            wanted.iter().any(|candidate| lower == *candidate)
        })
        .cloned()
}

struct Columns<'a> {
    emb: &'a str,
    title: Option<&'a str>,
    text: Option<&'a str>,
    id: Option<&'a str>,
}

fn append_batch(
    batch: &RecordBatch,
    columns: &Columns<'_>,
    limit: usize,
    rows: &mut Vec<Row>,
    skipped: &mut usize,
) -> Result<(), String> {
    let emb = batch
        .column_by_name(columns.emb)
        .ok_or_else(|| format!("missing column {}", columns.emb))?;
    let titles = optional_strings(batch, columns.title)?;
    let texts = optional_strings(batch, columns.text)?;
    let ids = optional_strings(batch, columns.id)?;
    for row in 0..batch.num_rows() {
        if rows.len() >= limit {
            break;
        }
        let Some(vector) = floats_at(emb, row)? else {
            *skipped += 1;
            continue;
        };
        if vector.is_empty() {
            *skipped += 1;
            continue;
        }
        if let Some(previous) = rows.first() {
            if previous.vector.len() != vector.len() {
                return Err(format!(
                    "embedding width changed from {} to {}",
                    previous.vector.len(),
                    vector.len()
                ));
            }
        }
        rows.push(Row {
            source_id: string_at(ids.as_ref(), row),
            title: string_at(titles.as_ref(), row),
            text: string_at(texts.as_ref(), row),
            vector,
        });
    }
    Ok(())
}

fn optional_strings(
    batch: &RecordBatch,
    name: Option<&str>,
) -> Result<Option<arrow_array::ArrayRef>, String> {
    let Some(name) = name else {
        return Ok(None);
    };
    let Some(column) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let utf8 = arrow_cast::cast(column, &DataType::Utf8).map_err(|err| err.to_string())?;
    Ok(Some(utf8))
}

fn string_at(column: Option<&arrow_array::ArrayRef>, row: usize) -> String {
    let Some(column) = column else {
        return String::new();
    };
    let values = column.as_string::<i32>();
    if values.is_null(row) {
        String::new()
    } else {
        values.value(row).to_string()
    }
}

fn floats_at(column: &arrow_array::ArrayRef, row: usize) -> Result<Option<Vec<f32>>, String> {
    if column.is_null(row) {
        return Ok(None);
    }
    let values: arrow_array::ArrayRef = match column.data_type() {
        DataType::FixedSizeList(_, _) => column.as_fixed_size_list().value(row),
        DataType::List(_) => column.as_list::<i32>().value(row),
        DataType::LargeList(_) => column.as_list::<i64>().value(row),
        other => {
            return Err(format!(
                "embedding column is {other}, expected a list of floats"
            ))
        }
    };
    match values.data_type() {
        DataType::Float32 => Ok(Some(
            values
                .as_primitive::<arrow_array::types::Float32Type>()
                .values()
                .to_vec(),
        )),
        DataType::Float64 => Ok(Some(
            values
                .as_primitive::<arrow_array::types::Float64Type>()
                .values()
                .iter()
                .map(|value| *value as f32)
                .collect(),
        )),
        other => Err(format!(
            "embedding values are {other}, want float32 or float64"
        )),
    }
}

fn read_csv(
    path: &Path,
    delimiter: u8,
    limit: usize,
    rows: &mut Vec<Row>,
    on_event: &mut impl FnMut(Event),
) -> Result<(), String> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .from_path(path)
        .map_err(|err| err.to_string())?;
    let headers = reader.headers().map_err(|err| err.to_string())?.clone();
    let header_as_floats = parse_float_row(headers.iter());
    let header_is_data = header_as_floats.is_some();
    if let Some(vector) = header_as_floats {
        push_row(
            rows,
            Row {
                source_id: String::new(),
                title: String::new(),
                text: String::new(),
                vector,
            },
        )?;
    }
    let emb_idx = column_index(&headers, &["emb", "embedding", "embeddings", "vector"]);
    let title_idx = column_index(&headers, &["title", "name"]);
    let text_idx = column_index(
        &headers,
        &["text", "content", "overview", "body", "abstract"],
    );
    let id_idx = column_index(&headers, &["id", "wiki_id"]);
    if !header_is_data {
        on_event(Event::Log(match emb_idx {
            Some(_) => "csv embedding column found".into(),
            None => "csv has no embedding column; using numeric columns as the vector".into(),
        }));
    }
    for record in reader.records() {
        if rows.len() >= limit {
            break;
        }
        let record = record.map_err(|err| err.to_string())?;
        let vector = if let Some(idx) = emb_idx {
            let cell = record.get(idx).unwrap_or("").trim();
            parse_json_floats(cell)?
        } else if header_is_data {
            parse_float_row(record.iter())
                .ok_or_else(|| "csv row is not all numbers".to_string())?
        } else {
            let mut nums = Vec::new();
            for (idx, cell) in record.iter().enumerate() {
                if Some(idx) == title_idx || Some(idx) == text_idx || Some(idx) == id_idx {
                    continue;
                }
                if let Ok(value) = cell.trim().parse::<f32>() {
                    nums.push(value);
                }
            }
            if nums.is_empty() {
                return Err("csv row has no numeric embedding values".into());
            }
            nums
        };
        push_row(
            rows,
            Row {
                source_id: id_idx
                    .and_then(|idx| record.get(idx))
                    .unwrap_or("")
                    .to_string(),
                title: title_idx
                    .and_then(|idx| record.get(idx))
                    .unwrap_or("")
                    .to_string(),
                text: text_idx
                    .and_then(|idx| record.get(idx))
                    .unwrap_or("")
                    .to_string(),
                vector,
            },
        )?;
        if rows.len() % LOG_EVERY == 0 {
            on_event(Event::Progress {
                done: rows.len() as u64,
                total: limit as u64,
            });
        }
    }
    Ok(())
}

fn column_index(headers: &csv::StringRecord, wanted: &[&str]) -> Option<usize> {
    headers.iter().position(|name| {
        let lower = name.trim().to_ascii_lowercase();
        wanted.iter().any(|candidate| lower == *candidate)
    })
}

fn parse_float_row<'a>(cells: impl Iterator<Item = &'a str>) -> Option<Vec<f32>> {
    let mut values = Vec::new();
    for cell in cells {
        values.push(cell.trim().parse::<f32>().ok()?);
    }
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn parse_json_floats(cell: &str) -> Result<Vec<f32>, String> {
    let value: serde_json::Value = serde_json::from_str(cell)
        .map_err(|_| format!("embedding cell is not a JSON array: {cell}"))?;
    json_floats(&value).ok_or_else(|| format!("embedding cell is not a list of numbers: {cell}"))
}

fn json_floats(value: &serde_json::Value) -> Option<Vec<f32>> {
    let items = value.as_array()?;
    items
        .iter()
        .map(|item| item.as_f64().map(|number| number as f32))
        .collect()
}

fn read_jsonl(
    path: &Path,
    limit: usize,
    rows: &mut Vec<Row>,
    on_event: &mut impl FnMut(Event),
) -> Result<(), String> {
    let file = File::open(path).map_err(|err| err.to_string())?;
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        if rows.len() >= limit {
            break;
        }
        let line = line.map_err(|err| err.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|err| format!("line {}: {err}", line_no + 1))?;
        let vector = ["emb", "embedding", "embeddings", "vector"]
            .iter()
            .find_map(|key| value.get(*key).and_then(json_floats))
            .ok_or_else(|| format!("line {}: no emb, embedding, or vector array", line_no + 1))?;
        push_row(
            rows,
            Row {
                source_id: json_string(&value, &["id", "wiki_id"]),
                title: json_string(&value, &["title", "name"]),
                text: json_string(&value, &["text", "content", "overview", "body", "abstract"]),
                vector,
            },
        )?;
        if rows.len() % LOG_EVERY == 0 {
            on_event(Event::Progress {
                done: rows.len() as u64,
                total: limit as u64,
            });
        }
    }
    Ok(())
}

fn json_string(value: &serde_json::Value, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(|item| item.as_str()))
        .unwrap_or("")
        .to_string()
}

fn push_row(rows: &mut Vec<Row>, row: Row) -> Result<(), String> {
    if let Some(previous) = rows.first() {
        if previous.vector.len() != row.vector.len() {
            return Err(format!(
                "embedding width changed from {} to {}",
                previous.vector.len(),
                row.vector.len()
            ));
        }
    }
    rows.push(row);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn silent(_: Event) {}

    #[test]
    fn tilde_and_doubled_home_resolve_to_the_same_file() {
        let root = std::env::temp_dir().join(format!("vdb-ui-home-{}", std::process::id()));
        let home = root.join("alice");
        let file = home.join("Downloads").join("dataset.tsv");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "a\t1\n").unwrap();

        assert_eq!(
            resolve_with_home("~/Downloads/dataset.tsv", Some(&home)),
            file
        );
        assert_eq!(
            resolve_with_home("~/Users/alice/Downloads/dataset.tsv", Some(&home)),
            file
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn csv_round_trip_finds_the_neighbor() {
        let dir = std::env::temp_dir().join(format!("vdb-ui-csv-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("points.csv");
        fs::write(
            &path,
            "id,title,emb\n1,alpha,\"[1.0, 0.0, 0.0]\"\n2,beta,\"[0.9, 0.1, 0.0]\"\n3,far,\"[0.0, 0.0, 1.0]\"\n",
        )
        .unwrap();
        let loaded = load_dataset(
            &LoadOptions {
                location: path.to_string_lossy().into_owned(),
                limit: 10,
                metric: Metric::Cosine,
                index: "flat".into(),
            },
            silent,
        )
        .unwrap();
        assert_eq!(loaded.dim, 3);
        assert_eq!(loaded.labels.len(), 3);
        let hits = loaded.engine.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits[0].id, 0);
        assert_eq!(loaded.labels[hits[1].id as usize].title, "beta");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parquet_round_trip_finds_the_neighbor() {
        let dir = std::env::temp_dir().join(format!("vdb-ui-parquet-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("points.parquet");
        write_parquet(&path);
        let loaded = load_dataset(
            &LoadOptions {
                location: dir.to_string_lossy().into_owned(),
                limit: 10,
                metric: Metric::L2,
                index: "hnsw".into(),
            },
            silent,
        )
        .unwrap();
        assert_eq!(loaded.dim, 4);
        let hits = loaded.engine.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(loaded.labels[hits[0].id as usize].title, "near");
        let _ = fs::remove_dir_all(&dir);
    }

    fn write_parquet(path: &Path) {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let emb_field = Field::new("emb", DataType::FixedSizeList(item.clone(), 4), false);
        let schema = Arc::new(Schema::new(vec![
            emb_field,
            Field::new("title", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, true),
        ]));
        let values = Float32Array::from(vec![
            1.0, 0.0, 0.0, 0.0, // near
            0.0, 5.0, 0.0, 0.0, // far
        ]);
        let emb = FixedSizeListArray::new(item, 4, Arc::new(values), None);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(emb),
                Arc::new(StringArray::from(vec!["near", "far"])),
                Arc::new(StringArray::from(vec!["close point", "distant point"])),
            ],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
}
