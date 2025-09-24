import os
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

class GoogleDriveStorage(StorageBackend):
    """Google Drive storage backend - user specifies exact folder"""
    
    def __init__(self, folder_link: str = None, credentials_path: str = "./credentials.json"):
        self.credentials_path = credentials_path
        self.token_path = "./token.json"
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        
        # Extract folder ID from various Google Drive URL formats
        self.folder_id = self._extract_folder_id(folder_link) if folder_link else None
        self.service = None
        
        # Memory file names in the Drive folder
        self.file_mapping = {
            "tiered_memories": "anima_memories.json",
            "affinity_data": "anima_affinity.json", 
            "integrity_data": "anima_integrity.json",
            "metadata": "anima_metadata.json"
        }
        
        self._authenticate()
    
    def _extract_folder_id(self, drive_link: str) -> str:
        """Extract folder ID from various Google Drive URL formats"""
        
        # Handle different Google Drive URL formats:
        patterns = [
            # https://drive.google.com/drive/folders/1BxYc3DeFgHiJkLmNoPqRsTuVwXyZ
            r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',
            
            # https://drive.google.com/drive/u/0/folders/1BxYc3DeFgHiJkLmNoPqRsTuVwXyZ
            r'drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9_-]+)',
            
            # https://drive.google.com/open?id=1BxYc3DeFgHiJkLmNoPqRsTuVwXyZ
            r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
            
            # Just the ID itself
            r'^([a-zA-Z0-9_-]{28,})$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_link)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract folder ID from: {drive_link}")
    
    def _authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Google Drive credentials not found at {self.credentials_path}. "
                        "Please download from Google Cloud Console."
                    )
                
                flow = Flow.from_client_secrets_file(self.credentials_path, self.scopes)
                flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'  # For desktop apps
                
                auth_url, _ = flow.authorization_url(prompt='consent')
                print(f"Please visit this URL to authorize Anima: {auth_url}")
                auth_code = input("Enter the authorization code: ")
                
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
            
            # Save credentials for next time
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive authentication successful")
    
    def _verify_folder_access(self) -> bool:
        """Verify we can access the specified folder"""
        if not self.folder_id:
            return False
            
        try:
            folder = self.service.files().get(fileId=self.folder_id).execute()
            if folder.get('mimeType') == 'application/vnd.google-apps.folder':
                print(f"✅ Connected to folder: {folder.get('name')}")
                return True
            else:
                print(f"❌ {self.folder_id} is not a folder")
                return False
        except Exception as e:
            print(f"❌ Cannot access folder {self.folder_id}: {e}")
            return False
    
    def _find_file_in_folder(self, filename: str) -> Optional[str]:
        """Find a file by name in the specified folder"""
        if not self.folder_id:
            return None
            
        try:
            query = f"'{self.folder_id}' in parents and name='{filename}' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            
            if files:
                return files[0]['id']  # Return first match
            return None
        except Exception as e:
            print(f"Error finding file {filename}: {e}")
            return None
    
    def _upload_file_to_folder(self, filename: str, content: str) -> bool:
        """Upload or update a file in the specified folder"""
        if not self.folder_id:
            print("No folder ID specified")
            return False
        
        try:
            # Create temporary file
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Check if file already exists
            existing_file_id = self._find_file_in_folder(filename)
            
            media = MediaFileUpload(temp_path, mimetype='application/json')
            
            if existing_file_id:
                # Update existing file
                self.service.files().update(
                    fileId=existing_file_id,
                    media_body=media
                ).execute()
                print(f"✅ Updated {filename} in Google Drive")
            else:
                # Create new file
                file_metadata = {
                    'name': filename,
                    'parents': [self.folder_id]
                }
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"✅ Created {filename} in Google Drive")
            
            # Clean up temp file
            os.remove(temp_path)
            return True
            
        except Exception as e:
            print(f"Error uploading {filename}: {e}")
            return False
    
    def _download_file_content(self, filename: str) -> Optional[str]:
        """Download file content from the folder"""
        file_id = self._find_file_in_folder(filename)
        if not file_id:
            return None
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            content = fh.getvalue().decode('utf-8')
            return content
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    def save_tiered_memories(self, fleeting: Dict, core: Dict, long: Dict) -> bool:
        """Save tiered memories to Google Drive"""
        if not self._verify_folder_access():
            return False
        
        try:
            data = {
                "fleeting": {k: self._serialize_mem_object(v) for k, v in fleeting.items()},
                "core": {k: self._serialize_mem_object(v) for k, v in core.items()},
                "long": {k: self._serialize_mem_object(v) for k, v in long.items()},
                "saved_at": datetime.utcnow().isoformat(),
                "total_memories": len(fleeting) + len(core) + len(long)
            }
            
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return self._upload_file_to_folder(self.file_mapping["tiered_memories"], content)
            
        except Exception as e:
            print(f"Error saving tiered memories: {e}")
            return False
    
    def load_tiered_memories(self) -> Tuple[Dict, Dict, Dict]:
        """Load tiered memories from Google Drive"""
        if not self._verify_folder_access():
            return {}, {}, {}
        
        content = self._download_file_content(self.file_mapping["tiered_memories"])
        if not content:
            print("No existing memory file found - starting fresh")
            return {}, {}, {}
        
        try:
            data = json.loads(content)
            print(f"📥 Loaded {data.get('total_memories', 0)} memories from Google Drive")
            
            fleeting = {k: self._deserialize_mem_object(v) for k, v in data.get("fleeting", {}).items()}
            core = {k: self._deserialize_mem_object(v) for k, v in data.get("core", {}).items()}
            long = {k: self._deserialize_mem_object(v) for k, v in data.get("long", {}).items()}
            
            return fleeting, core, long
        except Exception as e:
            print(f"Error loading tiered memories: {e}")
            return {}, {}, {}
    
    def save_affinity_data(self, affinity_log: List, memory_graph: Dict, tag_clusters: Dict) -> bool:
        """Save affinity data to Google Drive"""
        if not self._verify_folder_access():
            return False
        
        try:
            # Convert sets to lists for JSON serialization
            serializable_clusters = {k: list(v) for k, v in tag_clusters.items()}
            
            data = {
                "affinity_log": affinity_log,
                "memory_graph": dict(memory_graph),
                "tag_clusters": serializable_clusters,
                "saved_at": datetime.utcnow().isoformat(),
                "total_mappings": len(affinity_log)
            }
            
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return self._upload_file_to_folder(self.file_mapping["affinity_data"], content)
            
        except Exception as e:
            print(f"Error saving affinity data: {e}")
            return False
    
    def load_affinity_data(self) -> Tuple[List, Dict, Dict]:
        """Load affinity data from Google Drive"""
        if not self._verify_folder_access():
            return [], {}, {}
        
        content = self._download_file_content(self.file_mapping["affinity_data"])
        if not content:
            return [], {}, {}
        
        try:
            data = json.loads(content)
            print(f"📥 Loaded {data.get('total_mappings', 0)} affinity mappings from Google Drive")
            
            affinity_log = data.get("affinity_log", [])
            memory_graph = data.get("memory_graph", {})
            tag_clusters = {k: set(v) for k, v in data.get("tag_clusters", {}).items()}
            
            return affinity_log, memory_graph, tag_clusters
        except Exception as e:
            print(f"Error loading affinity data: {e}")
            return [], {}, {}
    
    def save_integrity_data(self, memory_registry: Dict, evaluation_history: List,
                           pending_clarifications: Dict, bondholder_patterns: Dict) -> bool:
        """Save integrity data to Google Drive"""
        if not self._verify_folder_access():
            return False
        
        try:
            # Serialize complex objects (similar to JSON storage)
            serialized_history = []
            for eval_obj in evaluation_history:
                if hasattr(eval_obj, '__dict__'):
                    eval_dict = eval_obj.__dict__.copy()
                    if hasattr(eval_obj.status, 'value'):
                        eval_dict['status'] = eval_obj.status.value
                    if hasattr(eval_obj.priority, 'value'):
                        eval_dict['priority'] = eval_obj.priority.value
                    serialized_history.append(eval_dict)
                else:
                    serialized_history.append(eval_obj)
            
            data = {
                "memory_registry": memory_registry,
                "evaluation_history": serialized_history,
                "pending_clarifications": dict(pending_clarifications),
                "bondholder_patterns": bondholder_patterns,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return self._upload_file_to_folder(self.file_mapping["integrity_data"], content)
            
        except Exception as e:
            print(f"Error saving integrity data: {e}")
            return False
    
    def load_integrity_data(self) -> Tuple[Dict, List, Dict, Dict]:
        """Load integrity data from Google Drive"""
        if not self._verify_folder_access():
            return {}, [], {}, {}
        
        content = self._download_file_content(self.file_mapping["integrity_data"])
        if not content:
            return {}, [], {}, {}
        
        try:
            data = json.loads(content)
            print(f"📥 Loaded integrity data from Google Drive")
            
            memory_registry = data.get("memory_registry", {})
            evaluation_history = data.get("evaluation_history", [])
            pending_clarifications = data.get("pending_clarifications", {})
            bondholder_patterns = data.get("bondholder_patterns", {})
            
            return memory_registry, evaluation_history, pending_clarifications, bondholder_patterns
        except Exception as e:
            print(f"Error loading integrity data: {e}")
            return {}, [], {}, {}
    
    def _serialize_mem_object(self, mem_obj) -> Dict:
        """Convert Mem object to JSON-serializable dict"""
        return {
            "id": mem_obj.id,
            "ts": mem_obj.ts.isoformat(),
            "text": mem_obj.text,
            "emotion": mem_obj.emotion,
            "intensity": mem_obj.intensity,
            "tags": mem_obj.tags,
            "meta": mem_obj.meta,
            "score": mem_obj.score,
            "pinned": mem_obj.pinned
        }
    
    def _deserialize_mem_object(self, mem_dict: Dict):
        """Convert dict back to Mem object"""
        mem = Mem(
            id=mem_dict["id"],
            ts=datetime.fromisoformat(mem_dict["ts"]),
            text=mem_dict["text"],
            emotion=mem_dict["emotion"],
            intensity=mem_dict["intensity"],
            tags=mem_dict["tags"],
            meta=mem_dict["meta"],
            pinned=mem_dict["pinned"]
        )
        mem.score = mem_dict["score"]
        return mem
    
    def get_folder_info(self) -> Dict:
        """Get information about the connected folder"""
        if not self.folder_id or not self._verify_folder_access():
            return {"connected": False}
        
        try:
            folder = self.service.files().get(fileId=self.folder_id).execute()
            
            # Count files in folder
            query = f"'{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(q=query).execute()
            files = results.get('files', [])
            
            anima_files = [f for f in files if f['name'].startswith('anima_')]
            
            return {
                "connected": True,
                "folder_name": folder.get('name'),
                "folder_id": self.folder_id,
                "total_files": len(files),
                "anima_files": len(anima_files),
                "last_sync": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

# Usage Examples:
def setup_google_drive_storage():
    """How a user would set this up"""
    
    # User drops their Google Drive folder link
    folder_link = "https://drive.google.com/drive/folders/1BxYc3DeFgHiJkLmNoPqRsTuVwXyZ"
    
    # Initialize storage
    storage = GoogleDriveStorage(folder_link)
    
    # Create Anima with Google Drive storage
    anima = PersistentCompleteMemorySystem(
        anima_instance,
        bondholder="User",
        storage_backend=storage
    )
    
    # Check connection
    folder_info = storage.get_folder_info()
    print(f"Connected to: {folder_info['folder_name']}")
    
    return anima

# In Anima Voice Harness, add command:
def handle_drive_setup(self, folder_link: str):
    """Handle /drive setup command"""
    try:
        storage = GoogleDriveStorage(folder_link)
        self.anima.memory.storage = storage
        
        folder_info = storage.get_folder_info()
        if folder_info["connected"]:
            print(f"✅ Connected to Google Drive folder: {folder_info['folder_name']}")
            print(f"📁 Found {folder_info['anima_files']} Anima files")
            
            # Load existing data
            self.anima.memory._load_all_data()
            print("📥 Loaded existing memories from Google Drive")
        else:
            print("❌ Failed to connect to Google Drive folder")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")

# Voice command integration:
# /drive https://drive.google.com/drive/folders/1BxYc3DeFgHiJkLmNoPqRsTuVwXyZ