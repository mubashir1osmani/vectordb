//! Local page for loading an embedding file and searching it.
//!
//! The Elm source is `ui/src/Main.elm`. Recompile it with:
//! `cd ui && elm make src/Main.elm --output ../crates/vdb-ui/assets/main.js`

mod load;
mod server;

use std::env;

#[tokio::main]
async fn main() {
    let port = parse_port(env::args().skip(1));
    let addr = format!("127.0.0.1:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("could not listen on {addr}: {err}");
            std::process::exit(1);
        }
    };
    println!("vectordb ui");
    println!("  http://{addr}");
    println!("  paste a local file or folder, an https URL, or use the sample");
    axum::serve(listener, server::router())
        .await
        .expect("server");
}

fn parse_port(args: impl Iterator<Item = String>) -> u16 {
    let args: Vec<String> = args.collect();
    for pair in args.windows(2) {
        if pair[0] == "--port" {
            if let Ok(port) = pair[1].parse() {
                return port;
            }
        }
    }
    8090
}
