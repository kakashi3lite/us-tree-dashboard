"""Google Drive Integration Utilities.

Provides minimal wrapper for authenticating (OAuth2 user flow) and performing
basic file operations (list, upload, download). Secrets are NOT stored in code.

Environment Variables:
  GDRIVE_CREDS_DIR: Directory path containing credentials.json (client secrets) and token.json (stored after auth).

Setup Steps (manual once):
 1. Place OAuth client JSON (download from Google Cloud Console) at: $GDRIVE_CREDS_DIR/credentials.json
 2. Run: python -m src.gdrive_integration --authorize
 3. Follow browser flow; token stored at token.json for future runs.

Usage (programmatic):
  from src.gdrive_integration import DriveClient
  client = DriveClient()
  files = client.list_files(page_size=10)
"""
from __future__ import annotations

import os
import pathlib
from typing import List, Optional, Dict, Any

from google.auth.transport.requests import Request  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload  # type: ignore
import io

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

class DriveClient:
    def __init__(self, creds_dir: Optional[str] = None):
        self.creds_dir = pathlib.Path(creds_dir or os.getenv("GDRIVE_CREDS_DIR", "gdrive_creds"))
        self.creds_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_path = self.creds_dir / "credentials.json"
        self.token_path = self.creds_dir / "token.json"
        self.creds: Optional[Credentials] = None
        self._service = None

    def authorize(self, force: bool = False):
        """Authorize via OAuth flow if needed.

        force: If True, ignore existing token and re-run flow.
        """
        if not force and self.token_path.exists():
            self.creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if self.creds and self.creds.expired and self.creds.refresh_token:
            self.creds.refresh(Request())
        if not self.creds or not self.creds.valid:
            if not self.credentials_path.exists():
                raise FileNotFoundError(
                    f"Missing OAuth client file at {self.credentials_path}. "
                    "Download from Google Cloud Console and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
            self.creds = flow.run_local_server(port=0)
            with open(self.token_path, "w", encoding="utf-8") as token:
                token.write(self.creds.to_json())
        self._service = build("drive", "v3", credentials=self.creds)

    @property
    def service(self):  # lazy init
        if not self._service:
            self.authorize()
        return self._service

    def list_files(self, page_size: int = 20) -> List[Dict[str, Any]]:
        results = self.service.files().list(pageSize=page_size, fields="files(id, name, mimeType, modifiedTime)").execute()
        return results.get("files", [])

    def upload_file(self, file_path: str, mime_type: str = "application/octet-stream", folder_id: Optional[str] = None) -> str:
        metadata: Dict[str, Any] = {"name": pathlib.Path(file_path).name}
        if folder_id:
            metadata["parents"] = [folder_id]
        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        file = self.service.files().create(body=metadata, media_body=media, fields="id").execute()
        return file["id"]

    def download_file(self, file_id: str, destination: str):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.close()

def _cli():  # pragma: no cover - manual tool
    import argparse, json
    parser = argparse.ArgumentParser(description="Google Drive helper")
    parser.add_argument("--authorize", action="store_true", help="Run OAuth flow")
    parser.add_argument("--list", action="store_true", help="List files")
    parser.add_argument("--upload", type=str, help="Path to upload")
    parser.add_argument("--download", nargs=2, metavar=("FILE_ID", "DEST"), help="Download file")
    parser.add_argument("--folder", type=str, help="Folder ID for upload", default=None)
    args = parser.parse_args()
    client = DriveClient()
    if args.authorize:
        client.authorize(force=True)
        print("Authorized and token stored.")
    if args.list:
        print(json.dumps(client.list_files(), indent=2))
    if args.upload:
        file_id = client.upload_file(args.upload, folder_id=args.folder)
        print(f"Uploaded file id: {file_id}")
    if args.download:
        client.download_file(args.download[0], args.download[1])
        print("Downloaded.")

if __name__ == "__main__":  # pragma: no cover
    _cli()