"""Services package for the Memories Retrieval System."""

# Core services (Phase 1)
from app.services.parser import (
    parse_memories_json,
    parse_date_string,
    load_memories_from_file,
    iter_memory_entries,
)
from app.services.downloader import (
    DownloadResult,
    download_single_media,
    download_all_media,
    cleanup_temp_files,
)
from app.services.image_processor import (
    ProcessingResult,
    process_media_file,
    is_zip_file,
    extract_snapchat_layers,
    merge_snapchat_layers,
)
from app.services.dedup import (
    DeduplicationService,
    get_dedup_service,
)
from app.services.storage import (
    upload_image,
    check_image_exists,
    delete_image,
    get_storage_path,
    get_signed_url,
    MockStorageService,
    get_mock_storage,
)
from app.services.indexer import (
    init_database,
    save_image_metadata,
    get_image_metadata,
    get_all_images,
    image_exists,
    delete_image_metadata,
    get_image_count,
    save_master_index_json,
    load_master_index_json,
    update_vector_ids,
    update_caption,
    get_images_without_captions,
    get_images_without_embeddings,
    get_images_with_captions_without_embeddings,
    get_all_images_with_embeddings,
    get_embedding_stats,
    get_image_by_vector_ids,
)

# Placeholder services (Phase 2)
from app.services.captioner import (
    CaptionResult,
    CaptionerService,
    get_captioner_service,
)
from app.services.embedder import (
    EmbeddingResult,
    EmbedderService,
    get_embedder_service,
)
from app.services.faiss_store import (
    SearchResult,
    FAISSStore,
    get_faiss_store,
)

__all__ = [
    # Parser
    "parse_memories_json",
    "parse_date_string",
    "load_memories_from_file",
    "iter_memory_entries",
    # Downloader
    "DownloadResult",
    "download_single_media",
    "download_all_media",
    "cleanup_temp_files",
    # Image Processor
    "ProcessingResult",
    "process_media_file",
    "is_zip_file",
    "extract_snapchat_layers",
    "merge_snapchat_layers",
    # Dedup
    "DeduplicationService",
    "get_dedup_service",
    # Storage
    "upload_image",
    "check_image_exists",
    "delete_image",
    "get_storage_path",
    "get_signed_url",
    "MockStorageService",
    "get_mock_storage",
    # Indexer
    "init_database",
    "save_image_metadata",
    "get_image_metadata",
    "get_all_images",
    "image_exists",
    "delete_image_metadata",
    "get_image_count",
    "save_master_index_json",
    "load_master_index_json",
    "update_vector_ids",
    "update_caption",
    "get_images_without_captions",
    "get_images_without_embeddings",
    "get_images_with_captions_without_embeddings",
    "get_all_images_with_embeddings",
    "get_embedding_stats",
    "get_image_by_vector_ids",
    # Captioner (Phase 2)
    "CaptionResult",
    "CaptionerService",
    "get_captioner_service",
    # Embedder (Phase 2)
    "EmbeddingResult",
    "EmbedderService",
    "get_embedder_service",
    # FAISS (Phase 2)
    "SearchResult",
    "FAISSStore",
    "get_faiss_store",
]
