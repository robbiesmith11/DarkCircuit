use std::{
    env,
    process::{Child, Command},
    sync::Mutex,
    time::Duration,
};
use tauri::{App, Manager};

struct BackendProcess(Mutex<Option<Child>>);

fn main() {
    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app: &mut App| {
            // Start the backend from onedir PyInstaller output
            let app_dir = app.path().resolve(".", tauri::path::BaseDirectory::Resource)?;
            println!(
                "✅ Resource path: {}",
                app.path()
                    .resolve(".", tauri::path::BaseDirectory::Resource)
                    .unwrap()
                    .display()
            );
            let backend_path = app_dir.join("dist").join("backend_server").join("backend_server");

            let child = Command::new(backend_path)
                .spawn()
                .expect("Failed to start backend");

            // Store backend process
            let state = app.state::<BackendProcess>();
            *state.0.lock().unwrap() = Some(child);

            // Wait for the server to start before loading UI
            std::thread::sleep(Duration::from_secs(1));
            if let Some(window) = app.get_webview_window("main") {
                window
                    .eval("window.location.replace('http://127.0.0.1:8000');")
                    .expect("Failed to redirect to backend");
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let app_handle = window.app_handle();
                let backend = app_handle.state::<BackendProcess>();
                if let Some(mut child) = backend.0.lock().unwrap().take() {
                    let _ = child.kill();
                }
                std::process::exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}