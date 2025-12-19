# Neuro-Trends Suite

**Integrated Platform for Neurodegenerative Analysis & Social Intelligence**

A consolidated, hardened system combining advanced neuroscience modeling with real-time social media trend detection. Built for medical research, clinical decision support, and market intelligence.

---

## Architecture Overview

The suite has been **completely refactored** from a fragmented prototype into a unified, production-grade stack:

### Core Components
- **Core API** (`core_api/`): Unified FastAPI gateway serving both Neuro and Trends endpoints
- **Unified Dashboard** (`dashboard_app.py`): Premium Streamlit interface for all analysis modes
- **PostgreSQL Backend**: Mandatory persistence layer for all predictions and social data
- **Shared Libraries** (`shared/lib/`): Common utilities including PII scrubbing, logging, and database management

### Legacy Components (Deprecated)
The following directories are **legacy** and will be removed in future versions:
- `neurodegenerai/src/api/` -> Migrated to `core_api/src/api/v1/endpoints/neuro.py`
- `trend-detector/src/api/` -> Migrated to `core_api/src/api/v1/endpoints/trends.py`

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)

### Launch the Full Stack
```bash
# Start all services (API, Dashboard, PostgreSQL, pgAdmin)
docker-compose up --build
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Core API
export PYTHONPATH=$PYTHONPATH:.
python -m uvicorn core_api.src.api.main:app --host 0.0.0.0 --port 8000

# Run Dashboard (in separate terminal)
streamlit run dashboard_app.py --server.port 8501
```

---

## Features

### NeuroDegenerAI
- **Tabular Biomarker Analysis**: Multi-modal ensemble predictions using LightGBM/XGBoost
- **MRI Structural Analysis**: 3D CNN for brain imaging classification
- **EEG Time-Series Decoding**: Real-time neurological state detection
- **Medical Data Validation**: Enforced schemas for ADNI-compliant data ingestion
- **PII Protection**: Automatic scrubbing of sensitive patient information

### Trend Detector
- **Real-Time Social Monitoring**: Reddit/Twitter stream ingestion with exponential backoff
- **Topic Clustering**: BERTopic-powered semantic analysis
- **Trend Intelligence**: Growth rate tracking and anomaly detection
- **Semantic Search**: Vector-based social data retrieval

---

## Security & Privacy

### Implemented Safeguards
1. **PII Scrubber**: Automatic redaction of emails, SSNs, phone numbers, and DOBs
2. **Data Validation**: Medical-grade range checks for all biomarker inputs
3. **Structured Logging**: Service-tagged JSON logs with no sensitive data leakage
4. **Database Isolation**: PostgreSQL with connection pooling and prepared statements

### Environment Configuration
Create a `.env` file:
```bash
# Database
POSTGRES_USER=trends
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=trends

# Reddit API (Optional)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=neuro-trends-suite/1.0

# System
ENV=production
LOG_LEVEL=INFO
NEURO_DEMO_MODE=false
```

---

## API Endpoints

### Neuro Analysis
- `POST /v1/neuro/tabular` - Biomarker prediction
- `POST /v1/neuro/mri` - MRI volume analysis
- `POST /v1/neuro/eeg` - EEG state decoding

### Trend Detection
- `GET /v1/trends/top` - Top trending topics
- `POST /v1/trends/search` - Semantic search

### System
- `GET /health` - Health check with DB status

Full API documentation: http://127.0.0.1:8000/docs

---

## Testing & Validation

```bash
# Run all tests
pytest

# Verify Core API
curl http://127.0.0.1:8000/health

# Test PII Scrubber
python -c "from shared.lib.io_utils import PIIScrubber; s = PIIScrubber(); print(s.scrub('Contact: john@example.com'))"
```

---

## Database Schema

### Tables
- `patients` - Patient demographics (anonymized)
- `neuro_predictions` - All model predictions with metadata
- `social_posts` - Ingested social media data
- `trend_topics` - Detected trending topics with scores

### Migrations
```bash
# Auto-create tables on first run
docker-compose up

# Manual migration (if needed)
alembic upgrade head
```

---

## Troubleshooting

### Common Issues

**API won't start:**
```bash
# Check PostgreSQL is running
docker-compose ps

# View logs
docker-compose logs core-api
```

**Dashboard connection error:**
```bash
# Ensure Core API is accessible
curl http://127.0.0.1:8000/health

# Check CORE_API_URL in dashboard_app.py
```

**Import errors:**
```bash
# Verify PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
```

---

## License

MIT License - See LICENSE file for details

---

## Medical Disclaimer

This software is for **research and educational purposes only**. It is not FDA-approved and should not be used for clinical diagnosis without proper validation and regulatory approval.

---

**Built with:** FastAPI, Streamlit, PostgreSQL, PyTorch, Scikit-learn, BERTopic, PRAW
