# Project Review: Atlas AI Data Catalog

## 1. Project Introduction

Atlas is an AI-assisted data catalog and metadata management system. It gives an organization one searchable inventory for datasets, ownership, schema, quality, business meaning, lineage, and machine-learning signals.

## 2. Problem Statement

Datasets are often scattered across systems, poorly documented, and difficult to compare. Manual metadata work is slow, quality problems are discovered late, and analysts may not know which asset is most relevant.

## 3. Existing System

Traditional spreadsheets and folder-based inventories record only names and locations. They do not automatically profile columns, score quality, classify business domains, or identify semantically related datasets.

## 4. Proposed System

Atlas combines a searchable catalog with upload-time Pandas profiling, quality metrics, role-based access, and three real ML workflows: Decision Tree classification, K-Means clustering, and TF-IDF cosine similarity.

## 5. Objectives

- Register and discover datasets.
- Extract technical and business metadata.
- Measure completeness, uniqueness, validity, and consistency.
- Automate category prediction, clustering, and similarity search.
- Track ownership, lineage, roles, and audit activity.

## 6. System Architecture

The browser UI sends authenticated REST requests to FastAPI. SQLAlchemy persists catalog records in SQLite for the local demo or MySQL in deployment. Uploaded files remain in `backend/uploads`; trained models are stored in `backend/models`.

## 7. Technologies Used

React, JavaScript, CSS, FastAPI, Uvicorn, SQLAlchemy, Pydantic, JWT, Pandas, NumPy, scikit-learn, Joblib, SQLite/MySQL, and REST APIs.

## 8. ML Algorithms

The application uses a Decision Tree for supervised-style category prediction, K-Means for unsupervised structural grouping, and TF-IDF plus cosine similarity for semantic discovery.

## 9. Decision Tree Explanation

Features include row count, column count, numeric/text ratios, missing-value percentage, and file size. The model predicts categories such as Financial, Customer, Education, Sales, HR, Marketing, Healthcare, and Operations and returns the highest class probability as confidence.

## 10. K-Means Explanation

The same structural features are standardized with `StandardScaler` and grouped into a configurable number of clusters. Each group is given a readable cluster label and stored against its datasets.

## 11. Cosine Similarity Explanation

Each dataset becomes a text document containing its name, description, tags, domain, and column names. `TfidfVectorizer` converts the documents to vectors; cosine similarity ranks the nearest neighboring datasets.

## 12. Database Design

Users, datasets, metadata, dataset columns, classifications, clusters, data quality, lineage, similarity, and audit logs are modeled as related tables. Dataset records own their metadata, columns, quality report, and classification.

## 13. Module Description

- Authentication: register, login, JWT, roles.
- Catalog: search, filters, detail views, ownership.
- Metadata: business and technical metadata.
- Quality: four dimensions and overall trust score.
- ML: classification, clustering, similarity.
- Lineage: source-to-analytics relationships.
- Administration: users, statuses, and settings.

## 14. System Workflow

User uploads a supported file, the backend validates and stores it, Pandas reads it, metadata and quality are calculated, ML services enrich the record, and the UI shows the result on the catalog and detail pages.

## 15. Results

The seeded review workspace begins with eight realistic datasets, eight categories, cluster assignments, quality scores, lineage records, and demo accounts. A new CSV can be uploaded without manually entering AI results.

## 16. Advantages

Centralized discovery, measurable quality, transparent model signals, reduced metadata effort, role-based control, and a clear local demonstration path.

## 17. Limitations

The default demo uses a lightweight SQLite database, synchronous upload processing, small seed data, and heuristic labels for the classifier training set. Production deployment should use an async queue, stronger governance, and curated training labels.

## 18. Future Scope

Add database/cloud connectors, scheduled scans, data contracts, human classification review, model monitoring, SSO, lineage extraction from orchestration tools, and object storage.

## 19. Conclusion

Atlas demonstrates how metadata management and machine learning can work together: the catalog provides context, quality provides trust, and ML improves organization and discovery.

## Two-minute faculty presentation

“Good morning. Our project is Atlas, an AI Data Catalog and Metadata Management System. The problem we address is that organizational datasets are usually spread across systems and are not consistently documented. An analyst may see a file name, but not its business meaning, quality, owner, or related datasets.

Atlas provides a searchable catalog with role-based access for administrators, data stewards, and analysts. When a user uploads a CSV, XLSX, or JSON file, the FastAPI backend stores the file and Pandas automatically extracts row counts, column types, null counts, unique values, and numeric statistics. The system then calculates completeness, uniqueness, validity, consistency, and an overall quality score.

The main innovation is the ML layer. A real Decision Tree uses measurable dataset features to classify an asset into categories such as Financial or Customer. Standardized feature vectors are grouped by real K-Means clustering to reveal structural families. Finally, TF-IDF and cosine similarity compare the dataset description, tags, business domain, and column names to recommend related datasets.

The React dashboard makes these results visible through catalog tables, quality indicators, model confidence, clusters, similarity bars, and lineage. For the demo, I log in as Admin, open the Financial Ledger, show its metadata and quality report, inspect the Decision Tree classification and K-Means cluster, then upload a new sample CSV. The system performs the entire pipeline automatically and displays the enriched record. This connects traditional catalog governance with practical machine learning and makes data easier to trust and reuse.”
