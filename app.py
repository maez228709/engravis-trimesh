import os, io, json, math, tempfile, urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
import trimesh
import numpy as np

app = Flask(__name__)
CORS(app)

MAX_FILE_MB = 150
NEIGHBOR_THRESHOLD = 0.05  # metros — meshes a menos de 5cm se consideran vecinos

# ─── Utilidades geométricas ──────────────────────────────────

def orientation(normal_mean):
    """Clasifica la orientación predominante de un mesh por su normal media."""
    nx, ny, nz = abs(normal_mean[0]), abs(normal_mean[1]), abs(normal_mean[2])
    if nz > 0.7:
        return "horizontal"   # cubierta, fondo, plataforma
    elif ny > 0.7 or nx > 0.7:
        return "vertical"     # mamparo, cuaderna, plancha lateral
    else:
        return "curved"       # casco, bocina, tobera

def mesh_summary(name, mesh):
    """Extrae propiedades geométricas de un mesh de Trimesh."""
    try:
        area = float(mesh.area) if mesh.area else 0.0
    except:
        area = 0.0

    try:
        is_closed = bool(mesh.is_watertight)
        volume = float(mesh.volume) if is_closed else 0.0
    except:
        is_closed = False
        volume = 0.0

    try:
        centroid = mesh.centroid.tolist()
    except:
        centroid = [0, 0, 0]

    try:
        bounds = mesh.bounds
        bbox = {
            "x": round(float(bounds[1][0] - bounds[0][0]), 4),
            "y": round(float(bounds[1][1] - bounds[0][1]), 4),
            "z": round(float(bounds[1][2] - bounds[0][2]), 4)
        }
    except:
        bbox = {"x": 0, "y": 0, "z": 0}

    try:
        normals = mesh.face_normals
        normal_mean = normals.mean(axis=0)
        norm = np.linalg.norm(normal_mean)
        if norm > 0:
            normal_mean = normal_mean / norm
        orient = orientation(normal_mean)
    except:
        normal_mean = [0, 0, 1]
        orient = "unknown"

    # Dimensión dominante para clasificación
    dims = sorted([bbox["x"], bbox["y"], bbox["z"]], reverse=True)
    aspect_ratio = round(dims[0] / dims[1], 2) if dims[1] > 0.001 else 999

    return {
        "name": name,
        "area_m2": round(area, 4),
        "volume_m3": round(volume, 6),
        "is_closed": is_closed,
        "centroid": [round(c, 4) for c in centroid],
        "bbox": bbox,
        "orientation": orient,
        "aspect_ratio": aspect_ratio,
        "face_count": len(mesh.faces)
    }

def find_neighbors(summaries, threshold=NEIGHBOR_THRESHOLD):
    """Encuentra pares de meshes cuya distancia entre centroides < threshold."""
    neighbors = {s["name"]: [] for s in summaries}
    centroids = {s["name"]: np.array(s["centroid"]) for s in summaries}
    names = list(centroids.keys())

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            dist = np.linalg.norm(centroids[a] - centroids[b])
            if dist < threshold:
                neighbors[a].append(b)
                neighbors[b].append(a)

    return neighbors

def cluster_meshes(summaries, neighbors):
    """Agrupa meshes conectados en clusters por proximidad."""
    visited = set()
    clusters = []

    name_to_summary = {s["name"]: s for s in summaries}

    def bfs(start):
        cluster = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend([n for n in neighbors.get(node, []) if n not in visited])
        return cluster

    for s in summaries:
        if s["name"] not in visited:
            cluster_names = bfs(s["name"])
            cluster_summaries = [name_to_summary[n] for n in cluster_names]

            # Propiedades del cluster
            total_area = sum(c["area_m2"] for c in cluster_summaries)
            total_volume = sum(c["volume_m3"] for c in cluster_summaries)
            centroids = [c["centroid"] for c in cluster_summaries]
            centroid_mean = [
                round(sum(c[i] for c in centroids) / len(centroids), 4)
                for i in range(3)
            ]

            # Orientación predominante del cluster
            orient_votes = {}
            for c in cluster_summaries:
                o = c["orientation"]
                orient_votes[o] = orient_votes.get(o, 0) + 1
            dominant_orient = max(orient_votes, key=orient_votes.get)

            clusters.append({
                "cluster_id": len(clusters),
                "mesh_names": cluster_names,
                "mesh_count": len(cluster_names),
                "area_m2": round(total_area, 4),
                "volume_m3": round(total_volume, 6),
                "centroid": centroid_mean,
                "orientation": dominant_orient,
                "bbox_approx": {
                    "x": round(max(c["bbox"]["x"] for c in cluster_summaries), 4),
                    "y": round(max(c["bbox"]["y"] for c in cluster_summaries), 4),
                    "z": round(max(c["bbox"]["z"] for c in cluster_summaries), 4)
                }
            })

    return clusters

# ─── Endpoints ───────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "engravis-trimesh"})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Recibe: { "glb_url": "https://..." }
    Retorna: análisis geométrico completo del GLB
    """
    data = request.get_json(silent=True) or {}
    glb_url = data.get("glb_url")

    if not glb_url:
        return jsonify({"error": "glb_url requerida"}), 400

    # Descargar GLB
    try:
        with urllib.request.urlopen(glb_url, timeout=60) as resp:
            size_mb = int(resp.headers.get("Content-Length", 0)) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                return jsonify({"error": f"Archivo supera {MAX_FILE_MB}MB"}), 413
            glb_bytes = resp.read()
    except Exception as e:
        return jsonify({"error": f"No se pudo descargar el GLB: {str(e)}"}), 400

    # Cargar con Trimesh
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(glb_bytes)
            tmp_path = tmp.name

        scene = trimesh.load(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Error al cargar GLB: {str(e)}"}), 422

    # Extraer meshes individuales
    if isinstance(scene, trimesh.Scene):
        meshes = {}
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes[name] = geom
    elif isinstance(scene, trimesh.Trimesh):
        meshes = {"Mesh_000": scene}
    else:
        return jsonify({"error": "Formato GLB no soportado"}), 422

    if not meshes:
        return jsonify({"error": "El GLB no contiene geometría válida"}), 422

    # Analizar cada mesh
    summaries = []
    for name, mesh in meshes.items():
        try:
            summaries.append(mesh_summary(name, mesh))
        except Exception as e:
            summaries.append({"name": name, "error": str(e)})

    # Encontrar vecinos y clusters
    valid_summaries = [s for s in summaries if "error" not in s]
    neighbors = find_neighbors(valid_summaries, NEIGHBOR_THRESHOLD)
    clusters = cluster_meshes(valid_summaries, neighbors)

    # Estadísticas globales del modelo
    total_area = sum(s.get("area_m2", 0) for s in valid_summaries)
    bbox_all = {
        "x_min": min(s["centroid"][0] - s["bbox"]["x"]/2 for s in valid_summaries) if valid_summaries else 0,
        "x_max": max(s["centroid"][0] + s["bbox"]["x"]/2 for s in valid_summaries) if valid_summaries else 0,
        "y_min": min(s["centroid"][1] - s["bbox"]["y"]/2 for s in valid_summaries) if valid_summaries else 0,
        "y_max": max(s["centroid"][1] + s["bbox"]["y"]/2 for s in valid_summaries) if valid_summaries else 0,
        "z_min": min(s["centroid"][2] - s["bbox"]["z"]/2 for s in valid_summaries) if valid_summaries else 0,
        "z_max": max(s["centroid"][2] + s["bbox"]["z"]/2 for s in valid_summaries) if valid_summaries else 0,
    }

    return jsonify({
        "status": "ok",
        "mesh_count": len(summaries),
        "cluster_count": len(clusters),
        "total_area_m2": round(total_area, 4),
        "model_bbox": {
            "length": round(bbox_all["x_max"] - bbox_all["x_min"], 4),
            "beam":   round(bbox_all["y_max"] - bbox_all["y_min"], 4),
            "depth":  round(bbox_all["z_max"] - bbox_all["z_min"], 4)
        },
        "meshes": summaries,
        "clusters": clusters
    })

@app.route('/cluster-names', methods=['POST'])
def cluster_names():
    """
    Recibe el JSON de /analyze ya procesado (clusters)
    y propone nombres SWBS basados en geometría.
    Esto es una pre-clasificación antes de enviar a ARIA/Groq.
    """
    data = request.get_json(silent=True) or {}
    clusters = data.get("clusters", [])

    proposals = []
    for c in clusters:
        orient = c.get("orientation", "unknown")
        area = c.get("area_m2", 0)
        bbox = c.get("bbox_approx", {})
        centroid = c.get("centroid", [0,0,0])
        aspect = bbox.get("x", 1) / bbox.get("z", 1) if bbox.get("z", 0.001) > 0.001 else 1

        # Heurísticas de clasificación geométrica
        if orient == "horizontal" and area > 10:
            guess = "cubierta"
            swbs_hint = "300"
        elif orient == "horizontal" and area < 5 and centroid[2] < 0:
            guess = "fondo"
            swbs_hint = "100"
        elif orient == "vertical" and aspect > 3:
            guess = "mamparo_longitudinal"
            swbs_hint = "200"
        elif orient == "vertical" and aspect < 1.5:
            guess = "mamparo_transversal"
            swbs_hint = "200"
        elif orient == "curved":
            guess = "casco"
            swbs_hint = "100"
        elif area < 0.5:
            guess = "elemento_pequeño"
            swbs_hint = "500"
        else:
            guess = "estructura_general"
            swbs_hint = "100"

        proposals.append({
            "cluster_id": c["cluster_id"],
            "geometric_guess": guess,
            "swbs_hint": swbs_hint,
            "confidence": "low",  # ARIA debe confirmar con contexto semántico
            "mesh_names": c["mesh_names"],
            "area_m2": c["area_m2"],
            "centroid": c["centroid"]
        })

    return jsonify({"proposals": proposals})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
