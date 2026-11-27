# Atlas: AI Data Catalog and Metadata Management System

Atlas is a full-stack college project that combines a searchable data catalog with automated metadata extraction, data-quality analysis, classification, clustering, semantic similarity, lineage, and role-based access.

## What is included

- React-style responsive Atlas UI in the root `index.html`, `index.js`, and `index.css` files. It can be opened through a local static server without a frontend build step.
- FastAPI backend in `backend/app/main.py` with JWT authentication and SQLAlchemy persistence.
- SQLite by default for a zero-setup local demonstration; MySQL is supported by changing `DATABASE_URL`.
- Upload processing for CSV, XLSX, and JSON with Pandas.
- Real `DecisionTreeClassifier`, `KMeans`, `StandardScaler`, `TfidfVectorizer`, and `cosine_similarity` calls.
- Seeded accounts and eight catalog records so the faculty demo starts with a populated system.
- Sample CSV files in `sample_data/`.

## Run locally

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000` and Swagger is at `http://localhost:8000/docs`.

### Frontend

In a second terminal from the project root:

```powershell
python -m http.server 5500
```

Open `http://localhost:5500`. The UI calls `http://localhost:8000/api` by default. To use another API URL, set `localStorage.atlas_api` in the browser console before refreshing.

A Vite package manifest is also provided in `frontend/package.json` for a production React migration/build.

## Demo credentials

| Role | Email | Password |
| --- | --- | --- |
| Admin | admin@example.com | Admin@123 |
| Data Steward | steward@example.com | Steward@123 |
| Data Analyst | analyst@example.com | Analyst@123 |

## API surface

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`
- `GET /api/datasets`, `GET /api/datasets/{id}`, `POST /api/datasets`, `PUT /api/datasets/{id}`, `DELETE /api/datasets/{id}`
- `GET /api/metadata/{dataset_id}`, `PUT /api/metadata/{dataset_id}`
- `POST /api/ml/classify/{dataset_id}`, `POST /api/ml/cluster`, `GET /api/datasets/{dataset_id}/similar`
- `GET /api/datasets/{dataset_id}/quality`
- `GET /api/dashboard/stats`
- `GET /api/users`, `PUT /api/users/{id}`
- `GET /api/lineage/{dataset_id}`, `POST /api/lineage`

## ML pipeline

1. Pandas reads the uploaded file.
2. Column metadata and quality dimensions are calculated.
3. Dataset features are normalized and sent to a scikit-learn Decision Tree classifier.
4. All dataset feature vectors are grouped with K-Means.
5. Dataset text is converted with TF-IDF and compared using cosine similarity.
6. Results are stored in the database and returned to the UI.

The generated Joblib files are placed in `backend/models/`.

## MySQL

Create a database and set an SQLAlchemy URL such as:

```env
DATABASE_URL=mysql+pymysql://catalog_user:password@localhost:3306/ai_data_catalog
JWT_SECRET=replace-me
UPLOAD_DIR=./uploads
CORS_ORIGINS=http://localhost:5500,http://localhost:5173
```

Start the API once; SQLAlchemy creates the tables. `database/schema.sql` contains the database bootstrap statement.

## Architecture

```text
React UI / static demo shell -> FastAPI REST API -> SQLAlchemy -> SQLite or MySQL
                                      |
                                      +-> Pandas metadata and quality service
                                      +-> Decision Tree / K-Means / TF-IDF ML services
                                      +-> backend/uploads and backend/models
```

## Demo workflow

Log in as Admin, open Overview, inspect the Financial Ledger or Customer Transactions record, open its metadata, quality, AI insight, similarity, and lineage tabs, then register a CSV from `sample_data/`. The backend automatically extracts columns, computes quality, runs all three ML paths, and returns the enriched dataset.

## Future enhancements

- Object storage for production file retention.
- Scheduled metadata refresh and source connectors.
- Human review queue for classification overrides.
- Row-level data policies and SSO.
- Async job queue for very large uploads.
