# Migration Guide: Legacy → Hardened Architecture

This document outlines the architectural changes made during the system consolidation.

## Overview
The Neuro-Trends Suite has been transformed from a **distributed monolith** into a **unified, production-ready platform**.

---

## Key Changes

### 1. API Consolidation
**Before:**
- `neurodegenerai/src/api/main.py` (Port 9001)
- `trend-detector/src/api/main.py` (Port 9002)

**After:**
- `core_api/src/api/main.py` (Port 8000)
  - `/v1/neuro/*` endpoints
  - `/v1/trends/*` endpoints

### 2. UI Unification
**Before:**
- `neurodegenerai/src/app/streamlit_app.py` (Port 8501)
- `trend-detector/src/app/streamlit_app.py` (Port 8502)
- `hub_app.py` (Port 8503)

**After:**
- `dashboard_app.py` (Port 8501)
  - Integrated navigation
  - Unified session management

### 3. Database Migration
**Before:**
- SQLite files scattered across services
- No persistence for trend data

**After:**
- Centralized PostgreSQL (Port 5432)
- SQLAlchemy models in `shared/lib/database.py`
- Auto-migration on startup

### 4. Security Enhancements
**Added:**
- PII Scrubber (`shared/lib/io_utils.py`)
- Medical data validation (`neurodegenerai/src/data/adni_ingest.py`)
- Exponential backoff for Reddit API (`trend-detector/src/ingest/reddit_stream.py`)

---

## Migration Steps

### For Developers

1. **Update imports:**
   ```python
   # Old
   from neurodegenerai.src.api.schemas import PredictionRequest

   # New
   from core_api.src.api.v1.schemas import TabularPredictionRequest
   ```

2. **Update API calls:**
   ```python
   # Old
   requests.post("http://127.0.0.1:9001/predict/tabular", ...)

   # New
   requests.post("http://127.0.0.1:8000/v1/neuro/tabular", ...)
   ```

3. **Database connections:**
   ```python
   # Old
   db_url = "sqlite:///./trend-detector/trends.db"

   # New
   from shared.lib.database import get_db_manager
   db_manager = get_db_manager()
   session = db_manager.get_session()
   ```

### For Deployment

1. **Update docker-compose:**
   - Remove legacy service definitions
   - Use the new consolidated `docker-compose.yml`

2. **Environment variables:**
   - Add PostgreSQL credentials
   - Update `DB_URL` to PostgreSQL connection string

3. **Port mappings:**
   - Core API: 8000
   - Dashboard: 8501
   - PostgreSQL: 5432
   - pgAdmin: 5050

---

## Deprecated Files

The following files are **no longer used** and can be safely removed:

- `neurodegenerai/src/api/main.py`
- `trend-detector/src/api/main.py`
- `neurodegenerai/src/app/streamlit_app.py`
- `trend-detector/src/app/streamlit_app.py`
- `hub_app.py`
- `neurodegenerai/Dockerfile` (replaced by root `Dockerfile`)
- `trend-detector/Dockerfile` (replaced by root `Dockerfile`)

---

## Testing the Migration

```bash
# 1. Start the new stack
docker-compose up --build

# 2.# Verify Core API
curl http://127.0.0.1:8000/health

# Open Dashboard
open http://127.0.0.1:8501

# Login to pgAdmin at http://127.0.0.1:5050
# Email: admin@neurotrends.ai
# Password: admin
```

---

## Rollback Plan

If issues arise, you can temporarily revert by:

1. Checking out the previous commit
2. Using the old `docker-compose.yml`
3. Running legacy services on original ports

However, **the new architecture is recommended** for all future development.
