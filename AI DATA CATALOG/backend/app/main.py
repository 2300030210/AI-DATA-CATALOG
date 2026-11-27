from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'catalog.db'}")
JWT_SECRET = os.getenv("JWT_SECRET", "local-demo-secret-change-me")
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(40), default="DATA ANALYST")
    status: Mapped[str] = mapped_column(String(20), default="Active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(180), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    source: Mapped[str] = mapped_column(String(100), default="Uploaded")
    file_name: Mapped[str] = mapped_column(String(255), default="")
    file_type: Mapped[str] = mapped_column(String(20), default="CSV")
    file_path: Mapped[str] = mapped_column(String(500), default="")
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    category: Mapped[str] = mapped_column(String(80), default="Unclassified")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    owner: Mapped[Optional[User]] = relationship()
    metadata_record: Mapped[Optional[MetadataRecord]] = relationship(back_populates="dataset", uselist=False, cascade="all, delete-orphan")
    columns: Mapped[list[DatasetColumn]] = relationship(cascade="all, delete-orphan")
    quality: Mapped[Optional[DataQuality]] = relationship(uselist=False, cascade="all, delete-orphan")
    classification: Mapped[Optional[Classification]] = relationship(uselist=False, cascade="all, delete-orphan")


class MetadataRecord(Base):
    __tablename__ = "metadata"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), unique=True)
    business_description: Mapped[str] = mapped_column(Text, default="")
    business_domain: Mapped[str] = mapped_column(String(100), default="General")
    data_owner: Mapped[str] = mapped_column(String(120), default="")
    data_steward: Mapped[str] = mapped_column(String(120), default="")
    department: Mapped[str] = mapped_column(String(100), default="")
    sensitivity: Mapped[str] = mapped_column(String(30), default="Internal")
    tags: Mapped[str] = mapped_column(Text, default="")
    usage_description: Mapped[str] = mapped_column(Text, default="")
    dataset: Mapped[Dataset] = relationship(back_populates="metadata_record")


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    column_name: Mapped[str] = mapped_column(String(180))
    data_type: Mapped[str] = mapped_column(String(80))
    null_count: Mapped[int] = mapped_column(Integer, default=0)
    unique_count: Mapped[int] = mapped_column(Integer, default=0)
    min_value: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    max_value: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    mean_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Classification(Base):
    __tablename__ = "classifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), unique=True)
    predicted_category: Mapped[str] = mapped_column(String(80))
    confidence: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(80), default="Decision Tree")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ClusterRecord(Base):
    __tablename__ = "clusters"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    cluster_id: Mapped[int] = mapped_column(Integer)
    cluster_name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class DataQuality(Base):
    __tablename__ = "data_quality"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), unique=True)
    completeness: Mapped[float] = mapped_column(Float)
    uniqueness: Mapped[float] = mapped_column(Float)
    validity: Mapped[float] = mapped_column(Float)
    consistency: Mapped[float] = mapped_column(Float)
    overall_score: Mapped[float] = mapped_column(Float)


class Lineage(Base):
    __tablename__ = "data_lineage"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_dataset_id: Mapped[Optional[int]] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    target_dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    relationship_type: Mapped[str] = mapped_column(String(100))


class SimilarDataset(Base):
    __tablename__ = "similar_datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    similar_dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    similarity_score: Mapped[float] = mapped_column(Float)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100))
    dataset_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(engine)


class RegisterInput(BaseModel):
    name: str = Field(min_length=2)
    email: str
    password: str = Field(min_length=6)
    role: str = "DATA ANALYST"


class LoginInput(BaseModel):
    email: str
    password: str


class MetadataInput(BaseModel):
    business_description: str = ""
    business_domain: str = "General"
    data_owner: str = ""
    data_steward: str = ""
    department: str = ""
    sensitivity: str = "Internal"
    tags: str = ""
    usage_description: str = ""


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None


class LineageInput(BaseModel):
    source_dataset_id: Optional[int] = None
    target_dataset_id: int
    relationship_type: str = "transforms"


class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None


def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_token(user: User) -> str:
    payload = {"sub": str(user.id), "exp": datetime.now(timezone.utc) + timedelta(hours=12)}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(db_session)) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub", 0))
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.get(User, user_id)
    if not user or user.status != "Active":
        raise HTTPException(status_code=401, detail="User is inactive or missing")
    return user


def require_roles(*roles: str):
    def dependency(user: User = Depends(current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="This role cannot perform that action")
        return user
    return dependency


def serialize_user(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role, "status": user.status, "created_at": user.created_at.isoformat()}


def feature_vector(dataset: Dataset) -> list[float]:
    numeric = sum(1 for column in dataset.columns if any(token in column.data_type.lower() for token in ("int", "float", "double", "number")))
    missing = 0.0
    if dataset.row_count and dataset.column_count:
        missing = sum(c.null_count for c in dataset.columns) / (dataset.row_count * dataset.column_count)
    size_mb = dataset.file_size / (1024 * 1024) if dataset.file_size else 0
    return [float(dataset.row_count), float(dataset.column_count), numeric / max(dataset.column_count, 1), 1 - numeric / max(dataset.column_count, 1), missing, size_mb]


def extract_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    if suffix == ".xlsx":
        return pd.read_excel(file_path)
    raise ValueError("Only CSV, XLSX, and JSON files are supported")


def analyze_dataframe(df: pd.DataFrame) -> tuple[list[dict], dict]:
    columns = []
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing = int(df.isna().sum().sum())
    duplicate_rate = float(df.duplicated().mean()) if len(df) else 0
    numeric_cols = df.select_dtypes(include=np.number).columns
    for name in df.columns:
        series = df[name]
        item: dict[str, Any] = {"column_name": str(name), "data_type": str(series.dtype), "null_count": int(series.isna().sum()), "unique_count": int(series.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(series):
            item.update({"min_value": str(series.min()), "max_value": str(series.max()), "mean_value": round(float(series.mean()), 4) if not series.dropna().empty else None})
        else:
            item.update({"min_value": None, "max_value": None, "mean_value": None})
        columns.append(item)
    completeness = round((1 - missing / total_cells) * 100, 2)
    uniqueness = round((1 - duplicate_rate) * 100, 2)
    validity = round(100 - min(100, (missing / total_cells) * 35), 2)
    consistency = round(max(0, uniqueness * 0.65 + completeness * 0.35), 2)
    quality = {"completeness": completeness, "uniqueness": uniqueness, "validity": validity, "consistency": consistency, "overall_score": round(np.mean([completeness, uniqueness, validity, consistency]), 2)}
    return columns, quality


def text_for_dataset(dataset: Dataset) -> str:
    metadata = dataset.metadata_record
    tags = metadata.tags if metadata else ""
    domain = metadata.business_domain if metadata else ""
    columns = " ".join(column.column_name for column in dataset.columns)
    return f"{dataset.name} {dataset.description} {tags} {domain} {columns}".lower()


def classify_dataset(dataset: Dataset, db: Session) -> Classification:
    all_datasets = list(db.scalars(select(Dataset)).all())
    labels = []
    vectors = []
    keywords = {"Financial": "amount balance transaction account revenue invoice", "Healthcare": "patient diagnosis hospital medical health", "Education": "student grade cgpa attendance course school", "Customer": "customer age gender income loyalty", "Sales": "sale order product quantity revenue", "HR": "employee salary department attendance", "Marketing": "campaign lead click conversion", "Operations": "inventory shipment warehouse operational"}
    for item in all_datasets:
        haystack = text_for_dataset(item)
        scores = {category: sum(word in haystack for word in words.split()) for category, words in keywords.items()}
        labels.append(max(scores, key=scores.get) if max(scores.values()) else "Operations")
        vectors.append(feature_vector(item))
    categories = list(keywords)
    if len(set(labels)) < 2:
        labels[-1] = "Operations" if labels[-1] != "Operations" else "Customer"
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(np.array(vectors), labels)
    probabilities = model.predict_proba([feature_vector(dataset)])[0]
    prediction = model.predict([feature_vector(dataset)])[0]
    confidence = float(max(probabilities))
    model_path = MODEL_DIR / "decision_tree.joblib"
    joblib.dump(model, model_path)
    record = dataset.classification or Classification(dataset_id=dataset.id)
    record.predicted_category, record.confidence, record.model_name = prediction, round(confidence, 4), "Decision Tree"
    dataset.category = prediction
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def run_clustering(db: Session) -> list[dict]:
    datasets = list(db.scalars(select(Dataset)).all())
    if not datasets:
        return []
    matrix = np.array([feature_vector(item) for item in datasets])
    scaled = StandardScaler().fit_transform(matrix) if len(datasets) > 1 else matrix
    count = min(4, len(datasets))
    model = KMeans(n_clusters=count, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    names = ["Core Analytics", "Customer Intelligence", "Financial Operations", "Enterprise Data"]
    db.query(ClusterRecord).delete()
    result = []
    for dataset, label in zip(datasets, labels):
        record = ClusterRecord(dataset_id=dataset.id, cluster_id=int(label) + 1, cluster_name=names[int(label) % len(names)])
        db.add(record)
        result.append({"dataset_id": dataset.id, "cluster_id": record.cluster_id, "cluster_name": record.cluster_name})
    joblib.dump(model, MODEL_DIR / "kmeans.joblib")
    db.commit()
    return result


def run_similarity(dataset_id: int, db: Session) -> list[dict]:
    datasets = list(db.scalars(select(Dataset)).all())
    target = next((item for item in datasets if item.id == dataset_id), None)
    if not target or len(datasets) < 2:
        return []
    corpus = [text_for_dataset(item) for item in datasets]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    scores = cosine_similarity(matrix[datasets.index(target)], matrix).flatten()
    results = []
    for index, score in sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True):
        other = datasets[index]
        if other.id != target.id:
            results.append({"dataset_id": other.id, "dataset_name": other.name, "similarity_score": round(float(score), 4), "common_tags": [], "common_domain": ""})
    db.query(SimilarDataset).filter(SimilarDataset.dataset_id == dataset_id).delete()
    for item in results[:5]:
        db.add(SimilarDataset(dataset_id=dataset_id, similar_dataset_id=item["dataset_id"], similarity_score=item["similarity_score"]))
    db.commit()
    return results[:5]


def serialize_dataset(dataset: Dataset, db: Session) -> dict:
    metadata = dataset.metadata_record
    classification = dataset.classification
    quality = dataset.quality
    cluster = db.scalar(select(ClusterRecord).where(ClusterRecord.dataset_id == dataset.id))
    return {"id": dataset.id, "name": dataset.name, "description": dataset.description, "owner": dataset.owner.name if dataset.owner else "Data Platform", "source": dataset.source, "file_name": dataset.file_name, "file_type": dataset.file_type, "file_size": dataset.file_size, "row_count": dataset.row_count, "column_count": dataset.column_count, "category": dataset.category, "created_at": dataset.created_at.isoformat(), "updated_at": dataset.updated_at.isoformat(), "metadata": metadata.__dict__ if metadata else None, "columns": [{k: v for k, v in column.__dict__.items() if not k.startswith("_")} for column in dataset.columns], "classification": {"predicted_category": classification.predicted_category, "confidence": classification.confidence, "model_name": classification.model_name} if classification else None, "quality": {k: getattr(quality, k) for k in ("completeness", "uniqueness", "validity", "consistency", "overall_score")} if quality else None, "cluster": {"cluster_id": cluster.cluster_id, "cluster_name": cluster.cluster_name} if cluster else None}


app = FastAPI(title="AI Data Catalog API", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:5500").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AI Data Catalog API", "database": DATABASE_URL.split(":")[0]}


@app.post("/api/auth/register")
def register(payload: RegisterInput, db: Session = Depends(db_session)):
    if db.scalar(select(User).where(User.email == payload.email.lower())):
        raise HTTPException(409, "Email is already registered")
    role = payload.role.upper() if payload.role.upper() in {"ADMIN", "DATA STEWARD", "DATA ANALYST"} else "DATA ANALYST"
    user = User(name=payload.name, email=payload.email.lower(), password_hash=pwd_context.hash(payload.password), role=role)
    db.add(user); db.commit(); db.refresh(user)
    return {"access_token": create_token(user), "token_type": "bearer", "user": serialize_user(user)}


@app.post("/api/auth/login")
def login(payload: LoginInput, db: Session = Depends(db_session)):
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if not user or not pwd_context.verify(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return {"access_token": create_token(user), "token_type": "bearer", "user": serialize_user(user)}


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return serialize_user(user)


@app.get("/api/datasets")
def list_datasets(search: str = "", category: str = "", source: str = "", db: Session = Depends(db_session), user: User = Depends(current_user)):
    query = select(Dataset).order_by(Dataset.updated_at.desc())
    datasets = list(db.scalars(query).all())
    term = search.lower().strip()
    output = []
    for item in datasets:
        haystack = text_for_dataset(item)
        if term and term not in haystack: continue
        if category and item.category != category: continue
        if source and item.source != source: continue
        output.append(serialize_dataset(item, db))
    return output


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(db_session), user: User = Depends(current_user)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset: raise HTTPException(404, "Dataset not found")
    return serialize_dataset(dataset, db)


@app.post("/api/datasets")
def create_dataset(file: UploadFile = File(...), name: str = Form(...), description: str = Form(""), source: str = Form("Uploaded"), domain: str = Form("General"), tags: str = Form(""), sensitivity: str = Form("Internal"), db: Session = Depends(db_session), user: User = Depends(current_user)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".json"}: raise HTTPException(400, "Only CSV, XLSX, and JSON files are allowed")
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename or "dataset")
    destination = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_name}"
    with destination.open("wb") as handle: shutil.copyfileobj(file.file, handle)
    if destination.stat().st_size > 25 * 1024 * 1024: destination.unlink(missing_ok=True); raise HTTPException(413, "File must be smaller than 25 MB")
    try: frame = extract_dataframe(destination); columns, quality = analyze_dataframe(frame)
    except Exception as exc: destination.unlink(missing_ok=True); raise HTTPException(400, f"Could not analyze file: {str(exc)[:120]}")
    dataset = Dataset(name=name, description=description, owner_id=user.id, source=source, file_name=safe_name, file_type=suffix[1:].upper(), file_path=str(destination), file_size=destination.stat().st_size, row_count=len(frame), column_count=len(frame.columns))
    db.add(dataset); db.flush()
    dataset.metadata_record = MetadataRecord(dataset_id=dataset.id, business_description=description, business_domain=domain, data_owner=user.name, data_steward=user.name, department="Data Platform", sensitivity=sensitivity, tags=tags)
    dataset.columns = [DatasetColumn(dataset_id=dataset.id, **column) for column in columns]
    dataset.quality = DataQuality(dataset_id=dataset.id, **quality)
    db.add(AuditLog(user_id=user.id, action="Created dataset", dataset_id=dataset.id)); db.commit(); db.refresh(dataset)
    classify_dataset(dataset, db); run_clustering(db); run_similarity(dataset.id, db)
    return serialize_dataset(dataset, db)


@app.put("/api/datasets/{dataset_id}")
def update_dataset(dataset_id: int, payload: DatasetUpdate, db: Session = Depends(db_session), user: User = Depends(require_roles("ADMIN", "DATA STEWARD"))):
    dataset = db.get(Dataset, dataset_id)
    if not dataset: raise HTTPException(404, "Dataset not found")
    for key, value in payload.model_dump(exclude_none=True).items(): setattr(dataset, key, value)
    db.add(AuditLog(user_id=user.id, action="Updated dataset", dataset_id=dataset.id)); db.commit(); db.refresh(dataset)
    return serialize_dataset(dataset, db)


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(db_session), user: User = Depends(require_roles("ADMIN"))):
    dataset = db.get(Dataset, dataset_id)
    if not dataset: raise HTTPException(404, "Dataset not found")
    db.delete(dataset); db.commit(); return {"message": "Dataset deleted"}


@app.get("/api/metadata/{dataset_id}")
def get_metadata(dataset_id: int, db: Session = Depends(db_session), user: User = Depends(current_user)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset or not dataset.metadata_record: raise HTTPException(404, "Metadata not found")
    return serialize_dataset(dataset, db)["metadata"]


@app.put("/api/metadata/{dataset_id}")
def update_metadata(dataset_id: int, payload: MetadataInput, db: Session = Depends(db_session), user: User = Depends(require_roles("ADMIN", "DATA STEWARD"))):
    dataset = db.get(Dataset, dataset_id)
    if not dataset: raise HTTPException(404, "Dataset not found")
    if not dataset.metadata_record: dataset.metadata_record = MetadataRecord(dataset_id=dataset.id)
    for key, value in payload.model_dump().items(): setattr(dataset.metadata_record, key, value)
    db.commit(); db.refresh(dataset)
    return serialize_dataset(dataset, db)["metadata"]


@app.post("/api/ml/classify/{dataset_id}")
def classify(dataset_id: int, db: Session = Depends(db_session), user: User = Depends(current_user)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset: raise HTTPException(404, "Dataset not found")
    record = classify_dataset(dataset, db)
    return {"dataset_id": dataset_id, "predicted_category": record.predicted_category, "confidence": record.confidence, "model": record.model_name}


@app.post("/api/ml/cluster")
def cluster(db: Session = Depends(db_session), user: User = Depends(current_user)):
    results = run_clustering(db)
    grouped: dict[int, dict] = {}
    for item in results:
        group = grouped.setdefault(item["cluster_id"], {"cluster_id": item["cluster_id"], "cluster_name": item["cluster_name"], "datasets": []})
        dataset = db.get(Dataset, item["dataset_id"]); group["datasets"].append({"id": dataset.id, "name": dataset.name})
    return [{**group, "cluster_size": len(group["datasets"])} for group in grouped.values()]


@app.get("/api/datasets/{dataset_id}/similar")
def similar(dataset_id: int, db: Session = Depends(db_session), user: User = Depends(current_user)):
    if not db.get(Dataset, dataset_id): raise HTTPException(404, "Dataset not found")
    return run_similarity(dataset_id, db)


@app.get("/api/datasets/{dataset_id}/quality")
def quality(dataset_id: int, db: Session = Depends(db_session), user: User = Depends(current_user)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset or not dataset.quality: raise HTTPException(404, "Quality report not found")
    return serialize_dataset(dataset, db)["quality"]


@app.get("/api/lineage/{dataset_id}")
def lineage(dataset_id: int, db: Session = Depends(db_session), user: User = Depends(current_user)):
    records = list(db.scalars(select(Lineage).where(or_(Lineage.source_dataset_id == dataset_id, Lineage.target_dataset_id == dataset_id))).all())
    return [{"id": item.id, "source_dataset_id": item.source_dataset_id, "target_dataset_id": item.target_dataset_id, "relationship_type": item.relationship_type} for item in records]


@app.post("/api/lineage")
def add_lineage(payload: LineageInput, db: Session = Depends(db_session), user: User = Depends(require_roles("ADMIN", "DATA STEWARD"))):
    record = Lineage(**payload.model_dump()); db.add(record); db.commit(); db.refresh(record); return {"id": record.id, **payload.model_dump()}


@app.get("/api/dashboard/stats")
def dashboard_stats(db: Session = Depends(db_session), user: User = Depends(current_user)):
    datasets = list(db.scalars(select(Dataset)).all()); quality = [item.quality.overall_score for item in datasets if item.quality]
    categories: dict[str, int] = {}; sources: dict[str, int] = {}; clusters: dict[str, int] = {}
    for item in datasets:
        categories[item.category] = categories.get(item.category, 0) + 1; sources[item.source] = sources.get(item.source, 0) + 1
        record = db.scalar(select(ClusterRecord).where(ClusterRecord.dataset_id == item.id))
        if record: clusters[record.cluster_name] = clusters.get(record.cluster_name, 0) + 1
    recent = [serialize_dataset(item, db) for item in datasets[:6]]
    return {"total_datasets": len(datasets), "total_sources": len(sources), "total_metadata_records": db.scalar(select(func.count(MetadataRecord.id))) or 0, "classified_datasets": sum(1 for item in datasets if item.classification), "average_quality": round(float(np.mean(quality)), 1) if quality else 0, "cluster_count": len(clusters), "categories": categories, "sources": sources, "clusters": clusters, "recent": recent}


@app.get("/api/users")
def users(db: Session = Depends(db_session), user: User = Depends(require_roles("ADMIN"))): return [serialize_user(item) for item in db.scalars(select(User).order_by(User.created_at)).all()]


@app.put("/api/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(db_session), user: User = Depends(require_roles("ADMIN"))):
    target = db.get(User, user_id)
    if not target: raise HTTPException(404, "User not found")
    for key, value in payload.model_dump(exclude_none=True).items(): setattr(target, key, value)
    db.commit(); return serialize_user(target)


def seed_database():
    db = SessionLocal()
    if db.scalar(select(User)): db.close(); return
    users = [User(name="Alex Morgan", email="admin@example.com", password_hash=pwd_context.hash("Admin@123"), role="ADMIN"), User(name="Jordan Lee", email="steward@example.com", password_hash=pwd_context.hash("Steward@123"), role="DATA STEWARD"), User(name="Sam Rivera", email="analyst@example.com", password_hash=pwd_context.hash("Analyst@123"), role="DATA ANALYST"), User(name="Priya Shah", email="analyst2@example.com", password_hash=pwd_context.hash("Analyst@123"), role="DATA ANALYST")]
    db.add_all(users); db.commit()
    sample_specs = [("Customer Transactions", "Customer purchase and loyalty behavior across regions.", "CRM / Warehouse", "Customer", "customer, loyalty, revenue", 12480, 8, 0.93), ("Financial Ledger", "Daily transaction ledger for finance operations.", "ERP / Finance", "Financial", "finance, ledger, transaction", 8420, 7, 0.96), ("Student Performance", "Academic performance, attendance, and course outcomes.", "SIS / Education", "Education", "students, grades, attendance", 3260, 7, 0.89), ("Employee Directory", "Workforce profile and organizational structure.", "HRIS", "HR", "employee, department, people", 1480, 7, 0.91), ("Campaign Engagement", "Marketing campaign delivery and conversion metrics.", "Marketing Cloud", "Marketing", "campaign, leads, conversion", 26400, 9, 0.86), ("Product Inventory", "Warehouse inventory, stock levels, and reorder signals.", "Operations DB", "Operations", "inventory, product, warehouse", 5120, 8, 0.88), ("Sales Orders", "Order lines and regional sales performance.", "ERP / Sales", "Sales", "sales, orders, revenue", 18900, 9, 0.92), ("Patient Visits", "De-identified appointment and care activity summary.", "Clinical System", "Healthcare", "patient, visits, diagnosis", 6750, 8, 0.94)]
    for index, (name, description, source, category, tags, rows, cols, score) in enumerate(sample_specs):
        dataset = Dataset(name=name, description=description, owner_id=users[index % 4].id, source=source, file_name=name.lower().replace(" ", "_") + ".csv", file_type="CSV", file_size=rows * cols * 18, row_count=rows, column_count=cols, category=category)
        db.add(dataset); db.flush(); dataset.metadata_record = MetadataRecord(dataset_id=dataset.id, business_description=description, business_domain=category, data_owner=users[index % 4].name, data_steward="Jordan Lee", department="Data & Analytics", sensitivity="Internal", tags=tags, usage_description="Approved for internal analytics and reporting.")
        for column_index in range(cols): db.add(DatasetColumn(dataset_id=dataset.id, column_name=["id", "name", "date", "amount", "status", "region", "score", "department", "source"][column_index], data_type="float64" if column_index in (3, 6) else "object", null_count=max(0, int(rows * (1 - score) / 3)), unique_count=int(rows * (0.6 + column_index / 20)), min_value="0" if column_index in (3, 6) else None, max_value="100000" if column_index in (3, 6) else None, mean_value=round(52.4 + column_index, 2) if column_index in (3, 6) else None))
        dataset.quality = DataQuality(dataset_id=dataset.id, completeness=score * 100, uniqueness=min(99, score * 98), validity=min(99, score * 101), consistency=score * 97, overall_score=score * 100)
        dataset.classification = Classification(dataset_id=dataset.id, predicted_category=category, confidence=round(min(0.99, score + 0.03), 2), model_name="Decision Tree")
    db.commit(); run_clustering(db)
    datasets = list(db.scalars(select(Dataset)).all())
    for index, dataset in enumerate(datasets): db.add(Lineage(source_dataset_id=None if index == 0 else datasets[index - 1].id, target_dataset_id=dataset.id, relationship_type="feeds" if index else "ingests"))
    db.commit(); db.close()


seed_database()
