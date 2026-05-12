# engravis-trimesh

Microservicio de análisis geométrico 3D para ENGRAVIS.
Analiza modelos GLB y extrae propiedades geométricas para clasificación semántica por ARIA.

## Endpoints

### GET /health
Verificación de estado del servicio.

### POST /analyze
Analiza un modelo GLB desde una URL pública.

**Request:**
```json
{ "glb_url": "https://ukcfzijdegclgavjtrug.supabase.co/storage/v1/object/public/models/..." }
```

**Response:**
```json
{
  "mesh_count": 234,
  "cluster_count": 23,
  "total_area_m2": 847.4,
  "model_bbox": { "length": 27.0, "beam": 6.5, "depth": 1.5 },
  "meshes": [...],
  "clusters": [...]
}
```

### POST /cluster-names
Pre-clasifica clusters por geometría (heurísticas).
Input: output de /analyze. Output: propuestas de nombre por cluster.

## Deploy en Render.com

1. Fork o push este repo a GitHub
2. Render.com → New Web Service → conectar repo
3. Runtime: Python, Plan: Free
4. Build: `pip install -r requirements.txt`
5. Start: `gunicorn app:app --workers 1 --timeout 120`

## Nota — Free tier
El servicio se duerme tras 15 min de inactividad.
El primer request puede tardar hasta 30 segundos.
