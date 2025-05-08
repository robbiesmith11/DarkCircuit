use std::{env, process::{Command, Child}, time::Duration, sync::{Arc, Mutex}, net::TcpStream, thread};
use tauri::{App, Manager};

fn main() {
  // Disable GPU compositing for Nouveau crash
  env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");

  // Store the backend process in an Arc<Mutex> so we can access it in the window close handler
  let backend_process = Arc::new(Mutex::new(None::<Child>));
  let backend_process_clone = backend_process.clone();

  tauri::Builder::default()
    .setup(move |app: &mut App| {
      // Start the Python backend
      let resource_path = app
        .path()
        .resolve("backend_server", tauri::path::BaseDirectory::Resource)
        .expect("Failed to resolve backend path");

      println!("Starting backend server at: {}", resource_path.display());

      // Start the backend process and store it
      let child = Command::new(&resource_path)
        .spawn()
        .expect("Failed to start backend");

      // Store the child process
      *backend_process.lock().unwrap() = Some(child);

      println!("Backend process started, waiting for server to be ready...");

      // Get the main window
      let window = app.get_webview_window("main").unwrap();

      // Create a loading message
      window
        .eval("document.body.innerHTML = '<div style=\"display: flex; justify-content: center; align-items: center; height: 100vh; font-family: Arial, sans-serif;\"><div style=\"text-align: center;\"><h2>Starting backend server...</h2><p>Please wait...</p></div></div>';")
        .unwrap();

      // Try to connect to the server with retries
      let max_retries = 30; // 30 seconds max wait time
      let server_url = "127.0.0.1:8000";

      let window_clone = window.clone();

      // Spawn a thread to check for server availability
      thread::spawn(move || {
        let mut connected = false;

        for i in 1..=max_retries {
          println!("Attempt {} to connect to server...", i);

          match TcpStream::connect(server_url) {
            Ok(_) => {
              println!("Server is ready!");
              connected = true;

              // Load the server URL into the Tauri window
              window_clone
                .eval("window.location.replace('http://127.0.0.1:8000');")
                .unwrap();

              break;
            }
            Err(e) => {
              println!("Server not ready yet: {}", e);
              thread::sleep(Duration::from_secs(1));
            }
          }
        }

        if !connected {
          // Server didn't come up, show error message
          window_clone
            .eval("document.body.innerHTML = '<div style=\"display: flex; justify-content: center; align-items: center; height: 100vh; font-family: Arial, sans-serif;\"><div style=\"text-align: center;\"><h2>Error</h2><p>Could not connect to backend server after multiple attempts.</p><p>Please restart the application.</p></div></div>';")
            .unwrap();
        }
      });

      // Set up a window close handler to terminate the backend process
      let window_clone = window.clone();
      window.on_window_event(move |event| {
        if let tauri::WindowEvent::CloseRequested { .. } = event {
          println!("Window close requested, terminating backend...");

          // Kill the backend process when the window is closed
          if let Some(mut child) = backend_process_clone.lock().unwrap().take() {
            match child.kill() {
              Ok(_) => println!("Backend process terminated successfully"),
              Err(e) => eprintln!("Failed to kill backend process: {}", e),
            }
          }

          // Allow some time for process to terminate
          thread::sleep(Duration::from_millis(500));

          // Close the window
          println!("Closing window");
          window_clone.close().unwrap();
        }
      });

      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}