# Memories Retrieval System

A multimodal image retrieval system for Snapchat Memories, allowing natural language search across your personal photo archive.

## 🎯 Features

### Phase 1 (Current)
- ✅ Upload `memories.json` from Snapchat export
- ✅ Download and process all media files
- ✅ Reconstruct images (merge main + overlay from ZIPs)
- ✅ Content-based deduplication (SHA256)
- ✅ Firebase Storage integration (or mock local storage)
- ✅ SQLite metadata database
- ✅ Master index management

### Phase 2 (Coming Soon)
- ⏳ Caption generation with Florence-2
- ⏳ CLIP embeddings for text and images
- ⏳ FAISS vector search
- ⏳ Natural language queries
- ⏳ Hybrid retrieval (70% text + 30% image similarity)

## 🏗️ Architecture

```
app/
├── main.py                # FastAPI bootstrap
├── config.py              # Paths, model names, constants
│
├── api/
│   ├── ingest.py          # Upload memories.json, trigger pipeline
│   ├── status.py          # Pipeline status
│   └── query.py           # Search images
│
├── services/
│   ├── parser.py          # Parse memories.json
│   ├── downloader.py      # Download media URLs to temp files
│   ├── image_processor.py # ZIP vs image, overlay merge
│   ├── dedup.py           # Hashing & deduplication
│   ├── storage.py         # Firebase upload logic
│   ├── captioner.py       # Florence-2 (Phase 2)
│   ├── embedder.py        # CLIP embeddings (Phase 2)
│   ├── faiss_store.py     # FAISS indices (Phase 2)
│   └── indexer.py         # Master index management
│
├── models/
│   └── schemas.py         # Pydantic models
│
└── utils/
    ├── hashing.py         # SHA256 computation
    ├── image_ops.py       # PIL operations
    └── logging.py         # Logging utilities
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn app.main:app --reload
```

### 3. Open API Docs

Navigate to `http://localhost:8000/docs` for interactive API documentation.

### 4. Upload Memories

```bash
curl -X POST "http://localhost:8000/ingest/" \
  -F "file=@memories_history.json"
```

### 5. Check Status

```bash
curl "http://localhost:8000/status/{job_id}"
```

## 📦 Data Storage

### Local (Persistent)
```
data/
├── db.sqlite              # Metadata + captions
├── master_index.json      # Backup index
├── mock_storage/          # Local images (dev only)
└── faiss/
    ├── text.index
    └── image.index
```

### Firebase Storage
Images are stored at: `images/{image_id}.jpg`

## 🔑 Key Principles

1. **Image Identity**: `image_id = SHA256(image_bytes)` - NEVER use filenames or timestamps
2. **Deduplication**: By content hash, not metadata
3. **No Local Image Persistence**: Images go to Firebase, only indices stay local
4. **Master Index is Truth**: Never re-scan storage to rebuild

## 🔧 Configuration

Set environment variables:

```bash
export MEMORIES_BASE_DIR="./data"
export MEMORIES_TEMP_DIR="/tmp/memories_processing"
export FIREBASE_BUCKET="your-project.appspot.com"
export FIREBASE_CREDENTIALS_PATH="./firebase-service-account.json"
export LOG_LEVEL="INFO"
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest/` | POST | Upload memories.json |
| `/ingest/test-parse` | POST | Test JSON parsing |
| `/status/{job_id}` | GET | Get pipeline status |
| `/status/` | GET | System status |
| `/query/search` | POST | Search images (Phase 2) |
| `/query/images` | GET | List all images |
| `/query/images/{id}` | GET | Get image details |

## 🧪 Development

### Mock Storage

For local development without Firebase, the system uses `MockStorageService` which saves images to `data/mock_storage/`.

### Testing

```bash
pytest tests/
```

## 📄 License

MIT
