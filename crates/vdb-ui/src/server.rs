//! HTTP API and the static Elm page. One load runs at a time.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use vectordb::Metric;

use crate::load::{self, Event, LoadOptions, Loaded};

const INDEX_HTML: &str = include_str!("../assets/index.html");
const APP_CSS: &str = include_str!("../assets/app.css");
const MAIN_JS: &str = include_str!("../assets/main.js");

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    phase: &'static str,
    stage: String,
    lines: Vec<String>,
    done: u64,
    total: Option<u64>,
    error: Option<String>,
    loaded: Option<Loaded>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::idle())),
        }
    }
}

impl Inner {
    fn idle() -> Self {
        Self {
            phase: "idle",
            stage: "idle".into(),
            lines: Vec::new(),
            done: 0,
            total: None,
            error: None,
            loaded: None,
        }
    }

    fn push(&mut self, line: String) {
        self.lines.push(line);
        if self.lines.len() > 80 {
            let extra = self.lines.len() - 80;
            self.lines.drain(0..extra);
        }
    }
}

pub fn router() -> Router {
    router_with(AppState::new())
}

pub fn router_with(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app.css", get(css))
        .route("/main.js", get(javascript))
        .route("/api/status", get(status))
        .route("/api/load", post(start_load))
        .route("/api/search", post(search))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS)
}

async fn javascript() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        MAIN_JS,
    )
}

#[derive(Serialize)]
struct StatusBody {
    phase: &'static str,
    stage: String,
    lines: Vec<String>,
    done: u64,
    total: Option<u64>,
    error: Option<String>,
    ready: Option<ReadyBody>,
}

#[derive(Serialize)]
struct ReadyBody {
    rows: usize,
    dim: usize,
    build_ms: u64,
    metric: String,
    index: String,
    preview: Vec<HitBody>,
}

#[derive(Serialize)]
struct HitBody {
    id: String,
    distance: Option<f32>,
    title: String,
    text: String,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

async fn status(State(state): State<AppState>) -> Json<StatusBody> {
    let inner = state.inner.lock().expect("status lock");
    Json(status_of(&inner))
}

fn status_of(inner: &Inner) -> StatusBody {
    let ready = inner.loaded.as_ref().map(|loaded| ReadyBody {
        rows: loaded.labels.len(),
        dim: loaded.dim,
        build_ms: loaded.build_ms,
        metric: loaded.metric.clone(),
        index: loaded.index.clone(),
        preview: loaded
            .labels
            .iter()
            .take(8)
            .map(|label| HitBody {
                id: label.id.to_string(),
                distance: None,
                title: label.title.clone(),
                text: label.text.clone(),
            })
            .collect(),
    });
    StatusBody {
        phase: inner.phase,
        stage: inner.stage.clone(),
        lines: inner.lines.clone(),
        done: inner.done,
        total: inner.total,
        error: inner.error.clone(),
        ready,
    }
}

#[derive(Deserialize)]
struct LoadRequest {
    location: String,
    limit: Option<u64>,
    metric: Option<String>,
    index: Option<String>,
}

async fn start_load(
    State(state): State<AppState>,
    Json(request): Json<LoadRequest>,
) -> Result<StatusCode, ApiError> {
    let limit = request.limit.unwrap_or(10_000);
    if limit == 0 || limit > 250_000 {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "limit must be from 1 to 250000".into(),
        });
    }
    let metric_name = request.metric.unwrap_or_else(|| "cosine".into());
    let metric = Metric::parse(&metric_name).ok_or_else(|| ApiError {
        status: StatusCode::BAD_REQUEST,
        message: format!("unknown metric `{metric_name}`"),
    })?;
    let index = request.index.unwrap_or_else(|| "hnsw".into());
    if index != "hnsw" && index != "flat" {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "index must be hnsw or flat".into(),
        });
    }
    if request.location.trim().is_empty() {
        return Err(ApiError {
            status: StatusCode::BAD_REQUEST,
            message: "location is empty".into(),
        });
    }
    {
        let mut inner = state.inner.lock().expect("load lock");
        if inner.phase == "loading" {
            return Err(ApiError {
                status: StatusCode::CONFLICT,
                message: "a dataset is already loading".into(),
            });
        }
        inner.phase = "loading";
        inner.stage = "resolve".into();
        inner.lines.clear();
        inner.done = 0;
        inner.total = Some(limit);
        inner.error = None;
        inner.loaded = None;
        inner.push(format!("location {}", request.location.trim()));
    }

    let worker = state.clone();
    let options = LoadOptions {
        location: request.location,
        limit: limit as usize,
        metric,
        index,
    };
    tokio::task::spawn_blocking(move || run_load(worker, options));
    Ok(StatusCode::ACCEPTED)
}

fn run_load(state: AppState, options: LoadOptions) {
    let state_for_events = state.clone();
    let result = load::load_dataset(&options, |event| apply_event(&state_for_events, event));
    let mut inner = state.inner.lock().expect("load result lock");
    match result {
        Ok(loaded) => {
            inner.phase = "ready";
            inner.stage = "ready".into();
            inner.done = loaded.labels.len() as u64;
            inner.total = Some(loaded.labels.len() as u64);
            inner.error = None;
            inner.loaded = Some(loaded);
        }
        Err(message) => {
            inner.phase = "failed";
            inner.stage = "failed".into();
            inner.error = Some(message.clone());
            inner.push(message);
            inner.loaded = None;
        }
    }
}

fn apply_event(state: &AppState, event: Event) {
    let mut inner = state.inner.lock().expect("event lock");
    match event {
        Event::Stage(stage) => inner.stage = stage.to_string(),
        Event::Log(line) => inner.push(line),
        Event::Progress { done, total } => {
            inner.done = done;
            inner.total = Some(total);
        }
    }
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    k: Option<usize>,
}

#[derive(Serialize)]
struct SearchResponse {
    hits: Vec<HitBody>,
}

async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let k = request.k.unwrap_or(10).clamp(1, 50);
    let inner = state.inner.lock().expect("search lock");
    let loaded = inner.loaded.as_ref().ok_or_else(|| ApiError {
        status: StatusCode::CONFLICT,
        message: "load a dataset before searching".into(),
    })?;
    let query = parse_query(&request.query, loaded).map_err(|message| ApiError {
        status: StatusCode::BAD_REQUEST,
        message,
    })?;
    let vector = match query {
        ParsedQuery::Id(id) => {
            loaded
                .vectors
                .get(id as usize)
                .cloned()
                .ok_or_else(|| ApiError {
                    status: StatusCode::BAD_REQUEST,
                    message: format!("no row with id {id}"),
                })?
        }
        ParsedQuery::Vector(vector) => vector,
    };
    let found = loaded.engine.search(&vector, k).map_err(|err| ApiError {
        status: StatusCode::BAD_REQUEST,
        message: err.to_string(),
    })?;
    let hits = found
        .into_iter()
        .filter_map(|hit| {
            loaded.labels.get(hit.id as usize).map(|label| HitBody {
                id: label.id.to_string(),
                distance: Some(hit.distance),
                title: label.title.clone(),
                text: label.text.clone(),
            })
        })
        .collect();
    Ok(Json(SearchResponse { hits }))
}

enum ParsedQuery {
    Id(u64),
    Vector(Vec<f32>),
}

fn parse_query(raw: &str, loaded: &Loaded) -> Result<ParsedQuery, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("query is empty".into());
    }
    let numeric = text.trim_matches(|ch| ch == '[' || ch == ']');
    let pieces: Vec<&str> = if numeric.contains(',') {
        numeric
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect()
    } else {
        numeric.split_whitespace().collect()
    };
    if pieces.len() > 1 {
        let mut vector = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let value = piece
                .parse::<f32>()
                .map_err(|_| format!("`{piece}` is not a number"))?;
            vector.push(value);
        }
        if vector.len() != loaded.dim {
            return Err(format!(
                "query has {} values, the index is {}-d",
                vector.len(),
                loaded.dim
            ));
        }
        return Ok(ParsedQuery::Vector(vector));
    }
    if let Ok(id) = text.parse::<u64>() {
        return Ok(ParsedQuery::Id(id));
    }
    Err("enter a row id, or a comma-separated vector".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_search_over_http() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        runtime.spawn(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            sender.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(listener, router()).await.unwrap();
        });
        let addr = receiver.recv().unwrap();
        let client = reqwest::blocking::Client::new();
        let base = format!("http://{addr}");
        let page = client.get(&base).send().unwrap().text().unwrap();
        assert!(page.contains("vectordb"));

        let started = client
            .post(format!("{base}/api/load"))
            .header("content-type", "application/json")
            .body(r#"{"location":"sample://clusters","limit":24,"metric":"cosine","index":"hnsw"}"#)
            .send()
            .unwrap();
        assert_eq!(started.status(), 202);

        let mut phase = String::new();
        for _ in 0..50 {
            let status: serde_json::Value = client
                .get(format!("{base}/api/status"))
                .send()
                .unwrap()
                .json()
                .unwrap();
            phase = status["phase"].as_str().unwrap_or("").to_string();
            if phase == "ready" || phase == "failed" {
                if phase == "failed" {
                    panic!("sample load failed: {status}");
                }
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert_eq!(phase, "ready");

        let found: serde_json::Value = client
            .post(format!("{base}/api/search"))
            .header("content-type", "application/json")
            .body(r#"{"query":"0","k":4}"#)
            .send()
            .unwrap()
            .json()
            .unwrap();
        let title = found["hits"][1]["title"].as_str().unwrap();
        assert!(title.starts_with("red"), "{title}");
    }
}
