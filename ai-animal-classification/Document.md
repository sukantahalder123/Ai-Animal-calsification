# AI Animal Image Classification - Web API

Welcome to the documentation for the AI Animal Image Classification project. This document outlines the project's objectives, architecture, folder structure, API endpoints, environment setup, and pseudocode for the core logic.

---

## Project Goals

The AI Animal Image Classification is designed to:

- Classify animal images (e.g., cow, dog, cat, bear) using a PyTorch MobileNetV2 model.
- Provide a Flask-based REST API for image upload and classification.
- Store predictions in a PostgreSQL database.
- Support history tracking and label-based search of past predictions.

---

## Folder Structure

```
ai-animal-classifier/
├── app.py                # Flask REST API server
├── predict.py            # Prediction logic using trained model
├── train.py              # PyTorch training script
├── model/
│   ├── model.pt          # Trained PyTorch model weights
│   └── class_names.txt   # Class names (labels)
├── dataset/              # Image dataset (class-wise folders)
├── uploads/              # Temporarily saved uploaded images
├── .env                  # PostgreSQL DB credentials
├── requirements.txt      # Python dependencies
└── DOCUMENTATION.md      # This documentation
```

---

## echnologies Used

- **Python 3.12+**
- **Flask** - Web server for REST API
- **PyTorch** - Deep learning framework
- **torchvision** - MobileNetV2 pre-trained model
- **PostgreSQL** - Relational database
- **psycopg2** - PostgreSQL client for Python
- **dotenv** - For loading `.env` environment variables

---

## API Endpoints

### 1. `GET /`

**Description**: Health check endpoint.

**Response:**
```json
"AI Animal Classifier API is running!"
```

---

### 2. `POST /upload`

**Description**: Upload an image, run prediction, and store result.

**Form-Data:**
- `image`: File (jpg/png)

**Success Response:**
```json
{
  "filename": "cow.jpg",
  "prediction": "cow"
}
```

**Error Response:**
```json
{ "error": "No image uploaded" }
```

---

### 3. `GET /history`

**Description**: Returns all predictions sorted by recent.

**Response:**
```json
[
  {
    "id": 1,
    "filename": "dog1.jpg",
    "label": "dog",
    "created_at": "2025-06-25T18:00:00"
  }
]
```

---

### 4. `GET /search?query=dog`

**Description**: Searches predictions by label.

**Query Param**:
- `query`: animal label (e.g., dog, cow)

**Response:**
```json
[
  {
    "id": 2,
    "filename": "dog3.jpg",
    "label": "dog",
    "image_url": "http://localhost:5000/uploads/dog3.jpg",
    "created_at": "2025-06-25T17:55:00"
  }
]
```

---

## Model Training Logic (train.py)

1. Load dataset from `dataset/` using `ImageFolder`
2. Apply transforms: resize, flip, rotation, normalization
3. Load `MobileNetV2`, replace classifier layer:
   ```python
   model.classifier[1] = nn.Linear(in_features, num_classes)
   ```
4. Use `Adam` optimizer and `CrossEntropyLoss`
5. Save:
   - Model → `model/model.pt`
   - Class labels → `model/class_names.txt`

---

## Pseudocode for Key APIs

### `/upload`
```
1. Check if image is present in request
2. Secure the filename, save to 'uploads/'
3. Call predict_image(filepath) to get label
4. Connect to PostgreSQL using psycopg2
5. Insert filename & label into 'predictions' table
6. Return filename and predicted label
```

### `/history`
```
1. Connect to PostgreSQL
2. Query all rows from 'predictions' table
3. Return as a JSON array sorted by created_at DESC
```

### `/search?query=label`
```
1. Read 'query' from URL params
2. Convert to lowercase for case-insensitive match
3. Query rows from 'predictions' where label = query
4. Return matched results with image URL
```

---

## Running the Project

### ➤ Development Mode

```bash
python app.py
```

## Environment Variables (`.env`)

```
POSTGRES_DB=yourDB
POSTGRES_USER=youruser
POSTGRES_PASSWORD=yourpass
POSTGRES_HOST=yourHost
POSTGRES_PORT=yourPort
```

---

## Database Schema (PostgreSQL)

```sql
CREATE TABLE predictions (
  id SERIAL PRIMARY KEY,
  filename TEXT,
  label TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Author

Created by **Sukanta Halder** — Built with ❤️ using Flask, PyTorch, and PostgreSQL.

---
