"""
Simple HTTP server to serve images from the data folder.
Run this to make images accessible to the HTML page.
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from pathlib import Path

class ImageHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="data", **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

if __name__ == "__main__":
    port = 8080
    server_address = ('', port)
    httpd = HTTPServer(server_address, ImageHandler)
    print(f"Image server running on http://localhost:{port}")
    print("Serving images from 'data' folder")
    print("Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
