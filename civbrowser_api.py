import os
import sys
import json
import re
import requests
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field

# Import modules from webui for direct access
from modules import script_callbacks, shared

class CivitaiAPISettings:
    def __init__(self):
        self.api_key = None
        self.settings_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "civitai_api_settings.json"
        )
        self.load_settings()
    
    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.api_key = settings.get('api_key')
                    print("Civitai API settings loaded")
        except Exception as e:
            print(f"Error loading Civitai API settings: {e}")
    
    def save_settings(self):
        """Save settings to JSON file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump({
                    'api_key': self.api_key
                }, f)
            print("Civitai API settings saved")
            return True
        except Exception as e:
            print(f"Error saving Civitai API settings: {e}")
            return False

civitai_settings = CivitaiAPISettings()

# Add the API route manually
def add_api_routes(app: FastAPI):
    """Add the Civitai Browser API routes"""
    
    # Model request validation
    class ModelCheckRequest(BaseModel):
        model_id: int = Field(..., description="Civitai Model ID")
        model_type: str = Field(..., description="Model type (checkpoint, lora, etc.)")
        version_id: Optional[int] = Field(None, description="Specific version ID (defaults to latest)")

    class ModelDownloadRequest(BaseModel):
        model_id: int = Field(..., description="Civitai Model ID")
        model_type: str = Field(..., description="Model type (checkpoint, lora, etc.)")
        version_id: Optional[int] = Field(None, description="Specific version ID (defaults to latest)")
        force: Optional[bool] = Field(False, description="Force download even if exists")

    class ModelResponse(BaseModel):
        exists: bool = Field(..., description="Whether the model exists locally")
        model_id: int = Field(..., description="Civitai Model ID")
        version_id: Optional[int] = Field(None, description="Version ID")
        filename: Optional[str] = Field(None, description="Original filename from Civitai")
        found_file: Optional[str] = Field(None, description="Actual filename found locally (may differ)")
        path: Optional[str] = Field(None, description="Full path if exists")

    class DownloadResponse(BaseModel):
        success: bool = Field(..., description="Download success status")
        message: str = Field(..., description="Status message")
        file_path: Optional[str] = Field(None, description="Path to downloaded file")
        model_id: int = Field(..., description="Civitai Model ID")
        version_id: Optional[int] = Field(None, description="Version ID")

    # Model deletion request
    class ModelDeleteRequest(BaseModel):
        model_type: str = Field(..., description="Model type (checkpoint, lora, etc.)")
        filename: str = Field(..., description="Filename to delete")
        empty_trash: bool = Field(True, description="Whether to empty trash after deletion")

    class APIKeyUpdate(BaseModel):
        api_key: str = Field(..., description="Civitai API Key")

    @app.post("/civitai/settings/api-key", tags=["Civitai Browser"])
    async def update_api_key(request: APIKeyUpdate):
        """Update the Civitai API key used for authenticated downloads"""
        try:
            # Update the API key in settings
            civitai_settings.api_key = request.api_key
            success = civitai_settings.save_settings()
            
            return {
                "success": success,
                "message": "API key updated successfully" if success else "Failed to save API key"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating API key: {str(e)}")
    
    @app.get("/civitai/settings/api-key-status", tags=["Civitai Browser"])
    async def get_api_key_status():
        """Check if a Civitai API key is configured"""
        try:
            has_key = civitai_settings.api_key is not None and civitai_settings.api_key.strip() != ""
            
            return {
                "has_api_key": has_key,
                "message": "API key is configured" if has_key else "No API key configured"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error checking API key status: {str(e)}")
        
    # Helper functions
    def get_civitai_api():
        """Get the CivitaiAPI instance"""
        try:
            # First try to import from original extension
            civbrowser_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sd-webui-civbrowser")
            if civbrowser_path not in sys.path:
                sys.path.append(civbrowser_path)
            from scripts.civitai_api import CivitaiAPI
            
            # Create API instance with API key if available
            api = CivitaiAPI()
            if civitai_settings.api_key:
                api.headers["Authorization"] = f"Bearer {civitai_settings.api_key}"
            
            return api
        except ImportError:
            # If not available, create minimal version
            class MinimalCivitaiAPI:
                def __init__(self):
                    self.base_url = "https://civitai.com/api/v1"
                    self.headers = {"Content-Type": "application/json"}
                    
                    # Add API key if available
                    if civitai_settings.api_key:
                        self.headers["Authorization"] = f"Bearer {civitai_settings.api_key}"
                    
                def get_model(self, model_id):
                    url = f"{self.base_url}/models/{model_id}"
                    response = requests.get(url, headers=self.headers)
                    return response.json()
            
            return MinimalCivitaiAPI()

    def get_model_folder(model_type):
        """Get the folder for a model type"""
        # Define folder mapping
        folders = {
            "checkpoint": os.path.join(shared.models_path, "Stable-diffusion"),
            "ckpt": os.path.join(shared.models_path, "Stable-diffusion"),
            "lora": os.path.join(shared.models_path, "Lora"),
            "locon": os.path.join(shared.models_path, "Lora"),
            "lycoris": os.path.join(shared.models_path, "LyCORIS"),
            "lyco": os.path.join(shared.models_path, "LyCORIS"),
            "embedding": shared.cmd_opts.embeddings_dir,
            "textualinversion": shared.cmd_opts.embeddings_dir,
            "ti": shared.cmd_opts.embeddings_dir,
            "hypernetwork": os.path.join(shared.models_path, "hypernetworks"),
            "vae": os.path.join(shared.models_path, "VAE"),
        }
        
        # Normalize model type
        model_type = model_type.lower()
        
        if model_type in folders:
            folder = folders[model_type]
            os.makedirs(folder, exist_ok=True)
            return folder
        
        # If not found, return None
        return None

    def find_model_file(model_filename, model_type):
        """Find a model file in the appropriate folder, handling various naming patterns"""
        folder = get_model_folder(model_type)
        if not folder:
            return None
        
        # Debug info
        print(f"Looking for model file: {model_filename} in {folder}")
        
        # Extract base name and extension
        name_without_ext, ext = os.path.splitext(model_filename)
        
        # Check if file exists with exact name
        full_path = os.path.join(folder, model_filename)
        if os.path.exists(full_path):
            print(f"Found exact match: {full_path}")
            return full_path
        
        # Check for common extensions if no extension or different extension
        if not ext:
            extensions = [".safetensors", ".ckpt", ".pt"]
            for test_ext in extensions:
                test_path = os.path.join(folder, name_without_ext + test_ext)
                if os.path.exists(test_path):
                    print(f"Found extension match: {test_path}")
                    return test_path
        
        # Check for files with ID suffix pattern (name_123456.ext)
        try:
            files_in_dir = os.listdir(folder)
        except Exception as e:
            print(f"Error listing files in directory: {e}")
            return None
        
        # First try to match exact name with any suffix
        suffix_pattern = f"{re.escape(name_without_ext)}_[0-9]+"
        if ext:
            suffix_pattern += f"\\{ext}$"
        
        suffix_regex = re.compile(suffix_pattern)
        
        for filename in files_in_dir:
            if suffix_regex.match(filename):
                print(f"Found ID suffix match: {filename}")
                return os.path.join(folder, filename)
        
        # Try more flexible matching (case insensitive, partial match)
        # This handles cases where the name might be slightly different
        name_lower = name_without_ext.lower()
        for filename in files_in_dir:
            file_lower = filename.lower()
            # If filename contains the model name and has the right extension
            if name_lower in file_lower and (not ext or file_lower.endswith(ext.lower())):
                print(f"Found partial name match with extension: {filename}")
                return os.path.join(folder, filename)
            # If filename contains the model name and has a common extension
            elif name_lower in file_lower and any(file_lower.endswith(e) for e in [".safetensors", ".ckpt", ".pt"]):
                print(f"Found partial name match with common extension: {filename}")
                return os.path.join(folder, filename)
        
        # If nothing is found, return None
        print(f"No matching file found for {model_filename}")
        return None

    def download_model_file(url, dest_path):
        """Download a file with progress reporting and rename to remove numeric suffix"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Modify URL to include token if available
            if civitai_settings.api_key:
                # Check if URL already has query parameters
                if '?' in url:
                    download_url = f"{url}&token={civitai_settings.api_key}"
                else:
                    download_url = f"{url}?token={civitai_settings.api_key}"
                print(f"Using token parameter in URL (token partially masked): {civitai_settings.api_key[:5]}...{civitai_settings.api_key[-5:] if len(civitai_settings.api_key) > 10 else ''}")
            else:
                download_url = url
                print("No API key configured, using URL without token")
            
            print(f"Downloading from URL: {download_url.replace(civitai_settings.api_key, 'TOKEN_HIDDEN') if civitai_settings.api_key else download_url}")
            
            # Download the file
            with requests.get(download_url, stream=True) as r:
                # Print response info for debugging
                print(f"Response status: {r.status_code}")
                print(f"Response headers: {dict(r.headers)}")
                
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                
                with open(dest_path, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Print progress
                            if total > 0:
                                percent = (downloaded / total) * 100
                                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total/(1024*1024):.1f}MB)")
                                sys.stdout.flush()
            
            print(f"\nDownload completed: {dest_path}")
            
            # Rename file to remove numeric suffix
            file_dir = os.path.dirname(dest_path)
            file_name = os.path.basename(dest_path)
            
            # Use regex to remove the numeric suffix before the extension
            import re
            new_file_name = re.sub(r'_\d+(\.\w+)$', r'\1', file_name)
            
            if new_file_name != file_name:
                new_dest_path = os.path.join(file_dir, new_file_name)
                os.rename(dest_path, new_dest_path)
                print(f"File renamed from {file_name} to {new_file_name}")
                return True, new_dest_path
            
            return True, dest_path
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"Authentication error 401: {str(e)}")
                print("This model requires proper authentication.")
                print("Possible solutions:")
                print("1. Check if your API key/token is correct")
                print("2. Make sure your API key has access to this model")
                print("3. This model might require you to be logged in with a Civitai account that has access")
                # Try to get the response body for more details
                try:
                    error_details = e.response.json()
                    print(f"Error details: {error_details}")
                except:
                    pass
            elif e.response.status_code == 403:
                print(f"Access forbidden (403): {str(e)}")
                print("This model might be restricted or private.")
            elif e.response.status_code == 404:
                print(f"Model not found (404): {str(e)}")
            else:
                print(f"HTTP error: {str(e)}")
            return False, None
        except Exception as e:
            print(f"Download error: {str(e)}")
            return False, None

    # Check if model exists endpoint
    @app.post("/civitai/exists", response_model=ModelResponse, tags=["Civitai Browser"])
    async def check_model_exists(request: ModelCheckRequest):
        """Check if a model is already downloaded"""
        try:
            # Get CivitaiAPI instance
            civitai_api = get_civitai_api()
            
            # Get model details from Civitai
            model_info = civitai_api.get_model(request.model_id)
            
            # Get or select version
            version = None
            version_id = request.version_id
            
            if version_id:
                # Find specific version
                for v in model_info.get("modelVersions", []):
                    if v.get("id") == version_id:
                        version = v
                        break
            else:
                # Use latest version
                if model_info.get("modelVersions"):
                    version = model_info["modelVersions"][0]
                    version_id = version.get("id")
            
            if not version:
                raise HTTPException(status_code=404, detail="Model version not found")
            
            # Find primary file
            file_info = None
            for file in version.get("files", []):
                if file.get("primary", False):
                    file_info = file
                    break
            
            # If no primary file, use first file
            if not file_info and version.get("files"):
                file_info = version["files"][0]
            
            if not file_info:
                raise HTTPException(status_code=404, detail="No files found for this model version")
            
            # Get filename
            filename = file_info.get("name")
            
            # Check if file exists
            file_path = find_model_file(filename, request.model_type)
            
            # Get actual filename if found
            found_file = os.path.basename(file_path) if file_path else None
            
            return {
                "exists": file_path is not None,
                "model_id": request.model_id,
                "version_id": version_id,
                "filename": filename,
                "found_file": found_file,
                "path": file_path
            }
        
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error checking model existence: {error_details}")
            raise HTTPException(status_code=500, detail=f"Error checking model: {str(e)}")

    # Download model endpoint
    @app.post("/civitai/download", response_model=DownloadResponse, tags=["Civitai Browser"])
    async def download_model(request: ModelDownloadRequest):
        """Download a model from Civitai if not already downloaded"""
        try:
            # First check if model exists
            check_request = ModelCheckRequest(
                model_id=request.model_id,
                model_type=request.model_type,
                version_id=request.version_id
            )
            
            # Get model existence info
            model_exists = await check_model_exists(check_request)
            
            # If model exists and not forcing download, return early
            if model_exists["exists"] and not request.force:
                return {
                    "success": True,
                    "message": f"Model already exists as {model_exists['found_file']}",
                    "file_path": model_exists["path"],
                    "model_id": request.model_id,
                    "version_id": model_exists["version_id"]
                }
            
            # Get model folder
            folder = get_model_folder(request.model_type)
            if not folder:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model type: {request.model_type}. Available types: checkpoint, lora, lycoris, embedding, hypernetwork, vae"
                )
            
            # Get CivitaiAPI instance
            civitai_api = get_civitai_api()
            
            # Get model details
            model_info = civitai_api.get_model(request.model_id)
            
            # Select version
            version = None
            if request.version_id:
                for v in model_info.get("modelVersions", []):
                    if v.get("id") == request.version_id:
                        version = v
                        break
            else:
                if model_info.get("modelVersions"):
                    version = model_info["modelVersions"][0]
            
            if not version:
                raise HTTPException(status_code=404, detail="Model version not found")
            
            # Find primary file
            file_info = None
            for file in version.get("files", []):
                if file.get("primary", False):
                    file_info = file
                    break
            
            # If no primary file, use first file
            if not file_info and version.get("files"):
                file_info = version["files"][0]
            
            if not file_info:
                raise HTTPException(status_code=404, detail="No files found for this model version")
            
            # Get filename and download URL
            filename = file_info.get("name")
            download_url = file_info.get("downloadUrl")
            
            if not download_url:
                raise HTTPException(status_code=404, detail="Download URL not found")
            
            # Create full path - add version ID to filename to avoid conflicts
            name_without_ext, ext = os.path.splitext(filename)
            versioned_filename = f"{name_without_ext}_{version.get('id')}{ext}"
            dest_path = os.path.join(folder, versioned_filename)
            
            # Download the file
            print(f"Downloading {filename} to {dest_path}")
            success = download_model_file(download_url, dest_path)
            
            if not success:
                raise HTTPException(status_code=500, detail="Download failed")
            
            # Refresh model list if needed (for checkpoints and VAEs)
            if request.model_type.lower() in ["checkpoint", "ckpt", "vae"]:
                from modules import sd_models
                sd_models.list_models()
            
            return {
                "success": True,
                "message": "Model downloaded successfully",
                "file_path": dest_path,
                "model_id": request.model_id,
                "version_id": version.get("id")
            }
        
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error downloading model: {error_details}")
            raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")

    # Debug endpoint to list files
    @app.post("/civitai/debug/files", tags=["Civitai Browser"])
    async def debug_model_files(
        model_type: str = Query(..., description="Model type (checkpoint, lora, etc.)"),
        search_term: Optional[str] = Query(None, description="Optional search term to filter files")
    ):
        """Debug endpoint to list all files of a given model type, optionally filtered by search term"""
        try:
            # Get model folder
            folder = get_model_folder(model_type)
            if not folder:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model type: {model_type}. Available types: checkpoint, lora, lycoris, embedding, hypernetwork, vae"
                )
            
            # List all files in the folder
            try:
                files = os.listdir(folder)
            except Exception as e:
                return {
                    "error": f"Could not list files in folder: {str(e)}",
                    "folder": folder,
                    "exists": os.path.exists(folder)
                }
            
            # Filter by search term if provided
            if search_term:
                search_term = search_term.lower()
                files = [f for f in files if search_term in f.lower()]
            
            # Return file details
            file_details = []
            for filename in files:
                full_path = os.path.join(folder, filename)
                try:
                    size = os.path.getsize(full_path)
                    modified = os.path.getmtime(full_path)
                    file_details.append({
                        "filename": filename,
                        "path": full_path,
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "modified": modified
                    })
                except Exception as e:
                    file_details.append({
                        "filename": filename,
                        "path": full_path,
                        "error": str(e)
                    })
            
            return {
                "folder": folder,
                "file_count": len(files),
                "files": file_details
            }
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

    @app.get("/civitai/storage", tags=["Civitai Browser"])
    async def get_storage_info():
        """Get storage information for the system and model directories"""
        import shutil
        import subprocess
        
        try:
            # Get system-wide storage info
            total, used, free = shutil.disk_usage("/")
            
            # Convert to more readable format
            total_gb = round(total / (1024**3), 2)
            used_gb = round(used / (1024**3), 2)
            free_gb = round(free / (1024**3), 2)
            used_percent = round((used / total) * 100, 2)
            
            # Get size of model directories
            model_sizes = {}
            for model_type in ["checkpoint", "lora", "lycoris", "embedding", "hypernetwork", "vae"]:
                folder = get_model_folder(model_type)
                if folder and os.path.exists(folder):
                    try:
                        # Use du command for more accurate directory size calculation
                        result = subprocess.run(
                            ["du", "-sh", folder], 
                            capture_output=True, 
                            text=True, 
                            check=True
                        )
                        # Parse the output (format: "10M /path/to/folder")
                        size_str = result.stdout.strip().split()[0]
                        model_sizes[model_type] = size_str
                    except (subprocess.SubprocessError, IndexError):
                        # Fallback to Python directory size calculation
                        size_bytes = 0
                        for dirpath, dirnames, filenames in os.walk(folder):
                            for f in filenames:
                                fp = os.path.join(dirpath, f)
                                if os.path.exists(fp):
                                    size_bytes += os.path.getsize(fp)
                        
                        # Convert to human-readable format
                        if size_bytes < 1024:
                            size_str = f"{size_bytes}B"
                        elif size_bytes < 1024**2:
                            size_str = f"{round(size_bytes/1024, 2)}KB"
                        elif size_bytes < 1024**3:
                            size_str = f"{round(size_bytes/(1024**2), 2)}MB"
                        else:
                            size_str = f"{round(size_bytes/(1024**3), 2)}GB"
                        
                        model_sizes[model_type] = size_str
            
            # Check if trash exists and get its size
            trash_dir = os.path.expanduser("~/.local/share/Trash")
            trash_exists = os.path.exists(trash_dir)
            trash_size = "0"
            
            if trash_exists:
                try:
                    result = subprocess.run(
                        ["du", "-sh", trash_dir], 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    trash_size = result.stdout.strip().split()[0]
                except (subprocess.SubprocessError, IndexError):
                    trash_size = "Unknown"
            
            return {
                "system": {
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "used_percent": used_percent
                },
                "model_directories": model_sizes,
                "trash": {
                    "exists": trash_exists,
                    "size": trash_size
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting storage info: {str(e)}")

    # Delete model endpoint
    @app.post("/civitai/delete", tags=["Civitai Browser"])
    async def delete_model(request: ModelDeleteRequest):
        """Delete a model file and optionally empty the trash"""
        try:
            # Get model folder
            folder = get_model_folder(request.model_type)
            if not folder:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid model type: {request.model_type}"
                )
            
            # Create full path
            file_path = os.path.join(folder, request.filename)
            
            # Check if file exists
            if not os.path.exists(file_path):
                # Try to find it with our more flexible matching
                file_path = find_model_file(request.filename, request.model_type)
                if not file_path:
                    raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")
            
            # Try moving to trash first (if available)
            import subprocess
            import shutil
            
            try:
                # First try using trash-cli if available
                subprocess.run(["trash-put", file_path], check=True)
                deletion_method = "trash-cli"
            except (subprocess.SubprocessError, FileNotFoundError):
                try:
                    # Try using gio trash if available
                    subprocess.run(["gio", "trash", file_path], check=True)
                    deletion_method = "gio"
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Fall back to direct removal
                    os.remove(file_path)
                    deletion_method = "direct"
            
            # Empty trash if requested
            trash_emptied = False
            if request.empty_trash:
                try:
                    # Remove trash directories
                    subprocess.run(["rm", "-rf", os.path.expanduser("~/.local/share/Trash/files")], check=True)
                    subprocess.run(["rm", "-rf", os.path.expanduser("~/.local/share/Trash/info")], check=True)
                    trash_emptied = True
                except subprocess.SubprocessError as e:
                    print(f"Error emptying trash: {str(e)}")
            
            # Refresh model list if needed
            if request.model_type.lower() in ["checkpoint", "ckpt", "vae"]:
                from modules import sd_models
                sd_models.list_models()
            
            return {
                "success": True,
                "deleted_file": os.path.basename(file_path),
                "deletion_method": deletion_method,
                "trash_emptied": trash_emptied,
                "path": file_path
            }
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

    class CleanupRequest(BaseModel):
        older_than_hours: int = Field(12, ge=1, le=24*90,
                                      description="Delete files older than this many hours (default: 12)")
        model_types: Optional[List[str]] = Field(
            None,
            description="Subset of model types to clean. Defaults to all known types."
        )
        dry_run: bool = Field(False, description="If true, only report what would be deleted")
        empty_trash: bool = Field(False, description="If true, empty trash after deletions")

    class CleanupResponse(BaseModel):
        success: bool
        dry_run: bool
        older_than_hours: int
        total_candidates: int
        total_deleted: int
        freed_bytes: int
        freed_mb: float
        details: List[Dict[str, Any]]
        trash_emptied: bool = False
        methods_tried: List[str] = []

    def _safe_delete_to_trash_or_remove(file_path: str) -> str:
        """
        Try to move file to trash (trash-cli or gio). Fallback to direct removal.
        Returns deletion method used.
        """
        import subprocess
        # 1) trash-cli
        try:
            subprocess.run(["trash-put", file_path], check=True)
            return "trash-cli"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        # 2) gio
        try:
            subprocess.run(["gio", "trash", file_path], check=True)
            return "gio"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        # 3) direct
        os.remove(file_path)
        return "direct"
    
    def _empty_trash_now() -> (bool, List[str]):
        import subprocess
        methods = []
        # trash-empty
        try:
            subprocess.run(["trash-empty"], check=True)
            methods.append("trash-empty command")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        # gio --empty
        try:
            subprocess.run(["gio", "trash", "--empty"], check=True)
            methods.append("gio trash --empty command")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        # direct rm fallback
        ok = True
        try:
            subprocess.run(["rm", "-rf", os.path.expanduser("~/.local/share/Trash/files")], check=True)
            subprocess.run(["rm", "-rf", os.path.expanduser("~/.local/share/Trash/info")], check=True)
            methods.append("direct rm -rf command")
        except subprocess.SubprocessError:
            ok = False
        # verify emptiness
        trash_empty = not os.path.exists(os.path.expanduser("~/.local/share/Trash/files")) and \
                      not os.path.exists(os.path.expanduser("~/.local/share/Trash/info"))
        return (ok and trash_empty), methods
    
    @app.post("/civitai/cleanup/older-than", response_model=CleanupResponse, tags=["Civitai Browser"])
    async def cleanup_models_older_than(request: CleanupRequest = Body(...)):
        """
        Delete files older than N hours inside known model folders.
        - Respects model_types (defaults to all known)
        - Dry-run support
        - Moves to trash when possible
        """
        try:
            known_types = ["checkpoint", "ckpt", "lora", "lycoris", "embedding", "hypernetwork", "vae"]
            target_types = request.model_types or known_types
    
            cutoff_epoch = time.time() - (request.older_than_hours * 3600)
            total_candidates = 0
            total_deleted = 0
            freed_bytes = 0
            details: List[Dict[str, Any]] = []
    
            for t in target_types:
                folder = get_model_folder(t)
                if not folder or not os.path.exists(folder):
                    details.append({
                        "model_type": t,
                        "folder": folder,
                        "error": "Folder not found or unsupported model_type"
                    })
                    continue
    
                try:
                    for entry in os.scandir(folder):
                        # Only regular files
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        try:
                            mtime = entry.stat(follow_symlinks=False).st_mtime
                            if mtime < cutoff_epoch:
                                size_b = entry.stat(follow_symlinks=False).st_size
                                total_candidates += 1
    
                                if request.dry_run:
                                    details.append({
                                        "model_type": t,
                                        "path": entry.path,
                                        "size_bytes": size_b,
                                        "last_modified": mtime,
                                        "action": "would_delete"
                                    })
                                else:
                                    method = _safe_delete_to_trash_or_remove(entry.path)
                                    total_deleted += 1
                                    freed_bytes += size_b
                                    details.append({
                                        "model_type": t,
                                        "path": entry.path,
                                        "size_bytes": size_b,
                                        "last_modified": mtime,
                                        "deleted_via": method
                                    })
                        except FileNotFoundError:
                            # File might disappear between listing and stat/delete
                            continue
                        except PermissionError as e:
                            details.append({
                                "model_type": t,
                                "path": entry.path,
                                "error": f"PermissionError: {e}"
                            })
                except Exception as e:
                    details.append({
                        "model_type": t,
                        "folder": folder,
                        "error": f"Error scanning folder: {e}"
                    })
    
            trash_emptied = False
            methods_tried: List[str] = []
            if not request.dry_run and request.empty_trash:
                trash_emptied, methods_tried = _empty_trash_now()
    
            return CleanupResponse(
                success=True,
                dry_run=request.dry_run,
                older_than_hours=request.older_than_hours,
                total_candidates=total_candidates,
                total_deleted=total_deleted,
                freed_bytes=freed_bytes,
                freed_mb=round(freed_bytes / (1024 * 1024), 2),
                details=details,
                trash_emptied=trash_emptied,
                methods_tried=methods_tried
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")
    
        # Empty trash endpoint
        @app.post("/civitai/empty-trash", tags=["Civitai Browser"])
        async def empty_trash():
            """Empty the system trash folder"""
            try:
                import subprocess
                
                # Try to empty trash using various methods
                trash_dir = os.path.expanduser("~/.local/share/Trash")
                trash_exists = os.path.exists(trash_dir)
                
                if not trash_exists:
                    return {
                        "success": True,
                        "message": "Trash directory does not exist or is already empty"
                    }
                
                methods_tried = []
                
                # Try trash-empty command if available
                try:
                    subprocess.run(["trash-empty"], check=True)
                    methods_tried.append("trash-empty command")
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                
                # Try gio trash --empty if available
                try:
                    subprocess.run(["gio", "trash", "--empty"], check=True)
                    methods_tried.append("gio trash --empty command")
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                
                # Direct removal of trash directories
                rm_success = True
                try:
                    subprocess.run(["rm", "-rf", os.path.expanduser("~/.local/share/Trash/files")], check=True)
                    subprocess.run(["rm", "-rf", os.path.expanduser("~/.local/share/Trash/info")], check=True)
                    methods_tried.append("direct rm -rf command")
                except subprocess.SubprocessError:
                    rm_success = False
                
                # Check if trash is now empty
                trash_empty = not os.path.exists(os.path.expanduser("~/.local/share/Trash/files")) and \
                              not os.path.exists(os.path.expanduser("~/.local/share/Trash/info"))
                
                return {
                    "success": trash_empty,
                    "methods_tried": methods_tried,
                    "trash_empty": trash_empty
                }
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error emptying trash: {str(e)}")
    
    # Return our app with routes added
    return app

# Function to register with the webui
def on_app_started(demo, app):
    """Register API routes when the webui starts"""
    try:
        add_api_routes(app)
        print("Civitai Browser API endpoints registered successfully")
        print("API documentation available at: /docs")
    except Exception as e:
        print(f"Error registering Civitai Browser API endpoints: {e}")

# Register our callback
script_callbacks.on_app_started(on_app_started)

# Check if the API is enabled
if not shared.cmd_opts.api:
    print("=" * 80)
    print("WARNING: API access is not enabled! To use the Civitai Browser API endpoints,")
    print("you need to start the webui with the '--api' flag:")
    print("    python launch.py --api")
    print("=" * 80)
