# Multimodal Memory System - Improved Version

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional but recommended:**
```bash
# For better OCR on stylized/multilingual text
pip install easyocr

# For image captioning (when OCR fails)
# Install Ollama and pull LLaVA
ollama pull llava
ollama pull llama3.1
```

### 2. File Structure

```
improved_code/
├── image_ingest_v2.py     # Image processing (OCR + captioning)
├── retrieval_v2.py        # Embedding-based semantic search
├── persistence.py         # SQLite storage
├── event_memory_v2.py     # Integrated memory system
├── voice_pipeline_v2.py   # Main application
├── requirements.txt       # Dependencies
└── evaluation_and_plan.md # Detailed analysis
```

### 3. Run the Voice Pipeline

```bash
# Make sure Ollama is running with required models
ollama serve

# In another terminal
python voice_pipeline_v2.py
```

### 4. Test Individual Components

```bash
# Test image ingestion
python image_ingest_v2.py /path/to/image.jpg

# Test embedding retrieval
python retrieval_v2.py

# Test persistence
python persistence.py

# Test integrated memory
python event_memory_v2.py
```

## Key Improvements Over Original

| Feature | Original | Improved |
|---------|----------|----------|
| OCR | Tesseract only | Tesseract + EasyOCR hybrid |
| Caption fallback | None | LLaVA VLM when OCR fails |
| Retrieval | Lexical (word overlap) | Embedding-based semantic |
| Persistence | None (in-memory) | SQLite with embeddings |
| Cross-modal linking | None | Temporal proximity detection |
| Response routing | Single path | Strategy-based (4 strategies) |

## Response Strategies

1. **DIRECT_READ**: High-confidence OCR → just read the text
2. **CAPTION_DESCRIBE**: OCR failed but caption available → describe image
3. **LLM_GROUND**: Complex query → use LLM with grounded context
4. **CLARIFY**: No context → ask user for clarification

## Configuration

Edit `Config` class in `voice_pipeline_v2.py`:

```python
class Config:
    # Audio
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 2
    AMPLITUDE_THRESHOLD = 0.02
    
    # Memory
    DB_PATH = "multimodal_memory.db"
    MAX_EVENTS = 100
    
    # LLM
    OLLAMA_URL = "http://127.0.0.1:11434"
    LLM_MODEL = "llama3.1"
    VLM_MODEL = "llava"
```

## Next Steps

See `evaluation_and_plan.md` for:
- Detailed analysis of current limitations
- Week-by-week improvement plan
- Advanced features roadmap (speaker diarization, memory consolidation)
