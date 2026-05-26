import os, io, json, math, tempfile, urllib.request
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import make_response
import trimesh
import numpy as np

app = Flask(__name__)
CORS(app, origins=['*'], methods=['GET','POST','OPTIONS'], allow_headers=['Content-Type','Authorization'])

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
        req = urllib.request.Request(glb_url, headers={"User-Agent": "Mozilla/5.0 (compatible; ENGRAVIS/1.0)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
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

# ── BASE DE DATOS NAVAL SWBS ─────────────────────────────────────────────────
# Basada en MIL-STD-1629 Ship Work Breakdown Structure + conocimiento naval
# Cada componente tiene: orientación típica, posición relativa (Z norm),
# rango de área, extensión longitudinal/transversal, y palabras clave.
# Z normalizado: 0.0 = fondo del buque, 1.0 = cubierta más alta

NAVAL_SWBS_DB = [
    # ── GRUPO 100 — ESTRUCTURA DEL CASCO ─────────────────────────────────
    {
        "swbs": "100", "codigo": "100-01", "nombre": "Fondo del Casco",
        "orientacion": ["horizontal"], "z_norm": (0.0, 0.2),
        "area_min": 15, "area_max": 999, "extension": "longitudinal",
        "aspect_xy_min": 3, "description": "Plancha de fondo, quilla plana, quilla de cajón"
    },
    {
        "swbs": "100", "codigo": "100-02", "nombre": "Quilla",
        "orientacion": ["vertical", "horizontal"], "z_norm": (0.0, 0.1),
        "area_min": 0.5, "area_max": 15, "extension": "longitudinal",
        "aspect_xy_min": 8, "description": "Barra de quilla, quilla de cajón central"
    },
    {
        "swbs": "100", "codigo": "100-03", "nombre": "Costado de Casco — Babor",
        "orientacion": ["vertical", "curved"], "z_norm": (0.1, 0.9),
        "area_min": 10, "area_max": 999, "extension": "longitudinal",
        "posicion_y": "negativo", "description": "Forro lateral babor, tracas de costado"
    },
    {
        "swbs": "100", "codigo": "100-04", "nombre": "Costado de Casco — Estribor",
        "orientacion": ["vertical", "curved"], "z_norm": (0.1, 0.9),
        "area_min": 10, "area_max": 999, "extension": "longitudinal",
        "posicion_y": "positivo", "description": "Forro lateral estribor, tracas de costado"
    },
    {
        "swbs": "100", "codigo": "100-05", "nombre": "Cuadernas / Varenga",
        "orientacion": ["vertical"], "z_norm": (0.0, 0.5),
        "area_min": 0.2, "area_max": 4, "extension": "transversal",
        "aspect_xy_min": 0, "aspect_xy_max": 1.5,
        "description": "Cuadernas transversales, varengas de fondo"
    },
    {
        "swbs": "100", "codigo": "100-06", "nombre": "Mamparo Transversal",
        "orientacion": ["vertical"], "z_norm": (0.0, 1.0),
        "area_min": 3, "area_max": 999, "extension": "transversal",
        "aspect_xy_max": 2, "description": "Mamparos estancos y estructurales transversales"
    },
    {
        "swbs": "100", "codigo": "100-07", "nombre": "Mamparo Longitudinal",
        "orientacion": ["vertical"], "z_norm": (0.0, 1.0),
        "area_min": 3, "area_max": 999, "extension": "longitudinal",
        "aspect_xy_min": 3, "description": "Mamparos longitudinales, crujía"
    },
    {
        "swbs": "100", "codigo": "100-08", "nombre": "Refuerzo Longitudinal / Eslora",
        "orientacion": ["horizontal", "vertical"], "z_norm": (0.0, 0.5),
        "area_min": 0.1, "area_max": 3, "extension": "longitudinal",
        "aspect_xy_min": 5, "description": "Esloras, palmejares, refuerzos longitudinales de fondo"
    },
    {
        "swbs": "100", "codigo": "100-09", "nombre": "Proa / Roda",
        "orientacion": ["vertical", "curved"], "z_norm": (0.2, 1.0),
        "area_min": 1, "area_max": 30, "extension": "transversal",
        "posicion_x": "extremo_proa", "description": "Estructura de proa, roda, bulbo de proa"
    },
    {
        "swbs": "100", "codigo": "100-10", "nombre": "Popa / Espejo",
        "orientacion": ["vertical", "curved"], "z_norm": (0.0, 1.0),
        "area_min": 1, "area_max": 30, "extension": "transversal",
        "posicion_x": "extremo_popa", "description": "Espejo de popa, codaste, estructura de popa"
    },
    {
        "swbs": "100", "codigo": "100-11", "nombre": "Doble Fondo",
        "orientacion": ["horizontal"], "z_norm": (0.0, 0.25),
        "area_min": 5, "area_max": 999, "extension": "longitudinal",
        "description": "Forro interior de doble fondo, tanques estructurales"
    },
    {
        "swbs": "100", "codigo": "100-12", "nombre": "Túnel de Hélice",
        "orientacion": ["curved"], "z_norm": (0.0, 0.3),
        "area_min": 1, "area_max": 20, "extension": "longitudinal",
        "posicion_x": "popa", "description": "Túnel de hélice, bocina, tubo de bocina"
    },
    # ── GRUPO 200 — PROPULSIÓN ────────────────────────────────────────────
    {
        "swbs": "200", "codigo": "200-01", "nombre": "Hélice",
        "orientacion": ["curved"], "z_norm": (0.0, 0.2),
        "area_min": 0.1, "area_max": 8, "extension": "ninguna",
        "posicion_x": "popa", "description": "Hélice propulsora, paso fijo o variable"
    },
    {
        "swbs": "200", "codigo": "200-02", "nombre": "Eje de Propulsión",
        "orientacion": ["horizontal"], "z_norm": (0.0, 0.2),
        "area_min": 0.05, "area_max": 2, "extension": "longitudinal",
        "aspect_xy_min": 6, "posicion_x": "popa",
        "description": "Línea de ejes, acoplamiento, bocina"
    },
    {
        "swbs": "200", "codigo": "200-03", "nombre": "Motor Principal",
        "orientacion": ["horizontal", "curved"], "z_norm": (0.05, 0.4),
        "area_min": 1, "area_max": 20,
        "posicion_x": "popa", "description": "Motor diésel principal, bloque motor"
    },
    {
        "swbs": "200", "codigo": "200-04", "nombre": "Chumaceras / Soportes de Eje",
        "orientacion": ["horizontal"], "z_norm": (0.0, 0.2),
        "area_min": 0.01, "area_max": 0.5, "extension": "ninguna",
        "description": "Chumaceras, soportes intermedios de línea de ejes"
    },
    # ── GRUPO 300 — GOBIERNO ──────────────────────────────────────────────
    {
        "swbs": "300", "codigo": "300-01", "nombre": "Timón",
        "orientacion": ["vertical"], "z_norm": (0.0, 0.6),
        "area_min": 0.3, "area_max": 15,
        "posicion_x": "extremo_popa", "aspect_xy_max": 1.5,
        "description": "Pala de timón, timón compensado o no compensado"
    },
    {
        "swbs": "300", "codigo": "300-02", "nombre": "Mecha de Timón",
        "orientacion": ["vertical"], "z_norm": (0.3, 0.8),
        "area_min": 0.01, "area_max": 0.5,
        "posicion_x": "extremo_popa", "description": "Mecha, pinzote, soporte de timón"
    },
    # ── GRUPO 400 — CUBIERTA ──────────────────────────────────────────────
    {
        "swbs": "400", "codigo": "400-01", "nombre": "Cubierta Principal",
        "orientacion": ["horizontal"], "z_norm": (0.75, 1.0),
        "area_min": 10, "area_max": 999, "extension": "longitudinal",
        "description": "Forro de cubierta principal, planchas de cubierta"
    },
    {
        "swbs": "400", "codigo": "400-02", "nombre": "Cubierta Intermedia",
        "orientacion": ["horizontal"], "z_norm": (0.35, 0.75),
        "area_min": 5, "area_max": 999, "extension": "longitudinal",
        "description": "Cubiertas interiores, sollados, cubiertas de bodega"
    },
    {
        "swbs": "400", "codigo": "400-03", "nombre": "Baos de Cubierta",
        "orientacion": ["horizontal"], "z_norm": (0.7, 1.0),
        "area_min": 0.1, "area_max": 3, "extension": "transversal",
        "aspect_xy_max": 0.5, "description": "Baos transversales de cubierta"
    },
    {
        "swbs": "400", "codigo": "400-04", "nombre": "Brazolas / Escotilla",
        "orientacion": ["vertical"], "z_norm": (0.75, 1.0),
        "area_min": 0.3, "area_max": 10,
        "description": "Brazolas de escotilla, marcos de escotilla"
    },
    {
        "swbs": "400", "codigo": "400-05", "nombre": "Tapa de Escotilla",
        "orientacion": ["horizontal"], "z_norm": (0.85, 1.0),
        "area_min": 0.5, "area_max": 30,
        "description": "Tapas de escotilla, paneles de cubierta removibles"
    },
    {
        "swbs": "400", "codigo": "400-06", "nombre": "Superestructura / Caseta",
        "orientacion": ["vertical", "horizontal"], "z_norm": (0.9, 1.0),
        "area_min": 2, "area_max": 100,
        "description": "Caseta de gobierno, superestructura, habilitación"
    },
    {
        "swbs": "400", "codigo": "400-07", "nombre": "Borda / Regala",
        "orientacion": ["vertical"], "z_norm": (0.8, 1.0),
        "area_min": 1, "area_max": 30, "extension": "longitudinal",
        "aspect_xy_min": 4, "description": "Borda, regala, pasamanos, balaustrada"
    },
    # ── GRUPO 500 — SISTEMAS AUXILIARES ──────────────────────────────────
    {
        "swbs": "500", "codigo": "500-01", "nombre": "Tanque de Combustible",
        "orientacion": ["horizontal", "vertical"], "z_norm": (0.0, 0.4),
        "area_min": 1, "area_max": 50,
        "description": "Tanques de combustible, tanques de servicio diario"
    },
    {
        "swbs": "500", "codigo": "500-02", "nombre": "Tanque de Agua Dulce",
        "orientacion": ["horizontal", "vertical"], "z_norm": (0.0, 0.5),
        "area_min": 0.5, "area_max": 30,
        "description": "Tanques de agua dulce, pique de proa"
    },
    {
        "swbs": "500", "codigo": "500-03", "nombre": "Tanque de Lastre",
        "orientacion": ["horizontal", "vertical"], "z_norm": (0.0, 0.3),
        "area_min": 2, "area_max": 100,
        "description": "Tanques de lastre, piques, doble fondo de lastre"
    },
    # ── GRUPO 600 — EQUIPOS DE CUBIERTA ──────────────────────────────────
    {
        "swbs": "600", "codigo": "600-01", "nombre": "Cabrestante / Molinete",
        "orientacion": ["curved", "horizontal"], "z_norm": (0.85, 1.0),
        "area_min": 0.1, "area_max": 3,
        "posicion_x": "proa", "description": "Molinete de ancla, cabrestante, chigre"
    },
    {
        "swbs": "600", "codigo": "600-02", "nombre": "Ancla / Escobén",
        "orientacion": ["curved"], "z_norm": (0.5, 1.0),
        "area_min": 0.01, "area_max": 1,
        "posicion_x": "proa", "description": "Ancla, escobén, caja de cadenas"
    },
    {
        "swbs": "600", "codigo": "600-03", "nombre": "Bita / Cornamusa",
        "orientacion": ["vertical"], "z_norm": (0.85, 1.0),
        "area_min": 0.01, "area_max": 0.3,
        "description": "Bitas de amarre, cornamusas, guías de cabo"
    },
    {
        "swbs": "600", "codigo": "600-04", "nombre": "Mástil / Palo",
        "orientacion": ["vertical"], "z_norm": (0.8, 1.0),
        "area_min": 0.05, "area_max": 2, "extension": "vertical",
        "aspect_xy_max": 0.3, "description": "Mástil principal, mástil de carga, palo"
    },
    {
        "swbs": "600", "codigo": "600-05", "nombre": "Grúa / Pórtico",
        "orientacion": ["vertical", "curved"], "z_norm": (0.8, 1.0),
        "area_min": 0.5, "area_max": 20,
        "description": "Grúa de carga, pórtico, pluma de carga"
    },
]

def preselect_candidates(cluster):
    """
    Filtra la DB naval y retorna los 5 componentes más probables
    basado en geometría: orientación, área, posición Z normalizada.
    """
    orient    = cluster.get("orientation", "unknown")
    area      = cluster.get("area_m2", 0)
    centroid  = cluster.get("centroid", [0, 0, 0])
    bbox      = cluster.get("bbox_approx", {})
    bx        = bbox.get("x", 0)
    by        = bbox.get("y", 0)
    bz        = bbox.get("z", 0)

    # Normalizar Z entre 0 y 1 usando límites aproximados del modelo
    # Se pasa z_min / z_max del modelo completo si está disponible
    z_min_model = cluster.get("z_min_model", -6)
    z_max_model = cluster.get("z_max_model", 2)
    z_range = max(z_max_model - z_min_model, 0.001)
    z_norm = (centroid[2] - z_min_model) / z_range

    # Aspect ratio longitudinal/transversal
    aspect_xy = bx / by if by > 0.001 else 1

    scores = []
    for comp in NAVAL_SWBS_DB:
        score = 0

        # Orientación
        if orient in comp.get("orientacion", []):
            score += 30

        # Posición Z normalizada
        z_range_comp = comp.get("z_norm", (0, 1))
        if z_range_comp[0] <= z_norm <= z_range_comp[1]:
            score += 25

        # Área
        a_min = comp.get("area_min", 0)
        a_max = comp.get("area_max", 9999)
        if a_min <= area <= a_max:
            score += 20

        # Extension / aspect
        ext = comp.get("extension", "")
        if ext == "longitudinal" and aspect_xy > 2:
            score += 10
        elif ext == "transversal" and aspect_xy < 1:
            score += 10

        scores.append((score, comp))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scores[:5]]


@app.route('/cluster-names', methods=['POST'])
def cluster_names():
    """
    Recibe clusters de /analyze-chunk procesados.
    1. Preselecciona candidatos SWBS por geometría (DB naval).
    2. Envía a Groq para clasificación semántica final con contexto naval.
    Retorna proposals con nombre, código SWBS y confianza.
    """
    import os, json, urllib.request

    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
    data     = request.get_json(silent=True) or {}
    clusters = data.get("clusters", [])

    if not clusters:
        return jsonify({"error": "clusters requeridos"}), 400

    # ── Paso 1: Pre-clasificación geométrica ─────────────────────────────
    pre_classified = []
    for c in clusters:
        candidates = preselect_candidates(c)
        pre_classified.append({
            "cluster_id":  c["cluster_id"],
            "orientation": c.get("orientation"),
            "area_m2":     round(c.get("area_m2", 0), 2),
            "centroid":    c.get("centroid", []),
            "mesh_count":  c.get("mesh_count", 0),
            "bbox":        c.get("bbox_approx", {}),
            "candidates":  [{"codigo": x["codigo"], "nombre": x["nombre"], "swbs": x["swbs"]} for x in candidates],
            "mesh_names":  c.get("mesh_names", [])[:3],
        })

    # ── Paso 2: Clasificación con Groq ───────────────────────────────────
    if not GROQ_KEY:
        # Sin Groq: retornar el candidato #1 como propuesta
        proposals = [{
            "cluster_id": p["cluster_id"],
            "nombre":     p["candidates"][0]["nombre"] if p["candidates"] else "Componente",
            "codigo":     p["candidates"][0]["codigo"] if p["candidates"] else "100-01",
            "swbs":       p["candidates"][0]["swbs"]   if p["candidates"] else "100",
            "confianza":  "media",
            "mesh_names": p["mesh_names"],
            "area_m2":    p["area_m2"],
        } for p in pre_classified]
        return jsonify({"proposals": proposals})

    # Construir prompt para Groq
    prompt_data = json.dumps(pre_classified, ensure_ascii=False)
    system_prompt = """Eres ARIA, asistente de ingeniería naval experta en clasificación SWBS (Ship Work Breakdown Structure).
Recibirás clusters de geometría 3D de un buque con candidatos preseleccionados.
Para cada cluster debes elegir el candidato más apropiado o proponer uno mejor si ninguno encaja.

CRITERIOS DE DECISIÓN:
- Usa orientación, área, posición Z normalizada (0=fondo, 1=cubierta), y aspect ratio.
- Considera la posición relativa entre clusters (elementos cercanos suelen ser del mismo sistema).
- Los mesh_names pueden dar pistas (GLTF_N no ayuda, pero "timón", "rudder", "helm" sí).
- Prioriza coherencia estructural del buque completo.

RESPONDE ÚNICAMENTE con JSON válido, sin texto adicional:
{
  "proposals": [
    {
      "cluster_id": <int>,
      "nombre": "<nombre en español>",
      "codigo": "<codigo SWBS como 100-01>",
      "swbs": "<grupo como 100>",
      "confianza": "<alta|media|baja>"
    }
  ]
}"""

    try:
        groq_body = json.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Clasifica estos {len(pre_classified)} clusters:\n{prompt_data}"}
            ],
            "max_tokens": 2000,
            "temperature": 0.2
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=groq_body,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {GROQ_KEY}"
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            groq_data = json.loads(resp.read())

        content = groq_data["choices"][0]["message"]["content"]
        # Limpiar markdown si viene con ```json
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())

        # Mezclar mesh_names y area del pre-classified
        id_map = {p["cluster_id"]: p for p in pre_classified}
        for prop in result.get("proposals", []):
            cid = prop.get("cluster_id")
            if cid in id_map:
                prop["mesh_names"] = id_map[cid]["mesh_names"]
                prop["area_m2"]    = id_map[cid]["area_m2"]

        return jsonify(result)

    except Exception as e:
        # Fallback a pre-clasificación geométrica
        proposals = [{
            "cluster_id": p["cluster_id"],
            "nombre":     p["candidates"][0]["nombre"] if p["candidates"] else "Componente",
            "codigo":     p["candidates"][0]["codigo"] if p["candidates"] else "100-01",
            "swbs":       p["candidates"][0]["swbs"]   if p["candidates"] else "100",
            "confianza":  "baja",
            "mesh_names": p["mesh_names"],
            "area_m2":    p["area_m2"],
            "error":      str(e)
        } for p in pre_classified]
        return jsonify({"proposals": proposals})


@app.route('/analyze-chunk', methods=['POST'])
def analyze_chunk():
    """
    Analiza un subconjunto de meshes del GLB (chunk).
    Recibe: glb_url, chunk_index, chunk_size (default 20)
    Retorna: summaries del chunk + total_meshes para saber cuántos chunks hay
    
    El cliente llama este endpoint múltiples veces con chunk_index=0,1,2...
    hasta cubrir todos los meshes. Luego combina y llama /classify.
    """
    data = request.get_json(silent=True) or {}
    glb_url    = data.get("glb_url")
    chunk_idx  = data.get("chunk_index", 0)
    chunk_size = data.get("chunk_size", 20)

    if not glb_url:
        return jsonify({"error": "glb_url requerida"}), 400

    # Descargar GLB
    try:
        req = urllib.request.Request(glb_url, headers={"User-Agent": "Mozilla/5.0 (compatible; ENGRAVIS/1.0)"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            glb_bytes = resp.read()
    except Exception as e:
        return jsonify({"error": f"No se pudo descargar: {str(e)}"}), 400

    # Cargar
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(glb_bytes)
            tmp_path = tmp.name
        scene = trimesh.load(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Error al cargar GLB: {str(e)}"}), 422

    # Extraer meshes
    if isinstance(scene, trimesh.Scene):
        all_meshes = {n: g for n, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh)}
    elif isinstance(scene, trimesh.Trimesh):
        all_meshes = {"Mesh_000": scene}
    else:
        return jsonify({"error": "Formato no soportado"}), 422

    mesh_names = list(all_meshes.keys())
    total      = len(mesh_names)

    # Calcular rango del chunk
    start = chunk_idx * chunk_size
    end   = min(start + chunk_size, total)

    if start >= total:
        return jsonify({
            "status":       "ok",
            "chunk_index":  chunk_idx,
            "chunk_size":   chunk_size,
            "total_meshes": total,
            "total_chunks": math.ceil(total / chunk_size),
            "is_last":      True,
            "summaries":    []
        })

    chunk_names  = mesh_names[start:end]
    summaries    = []

    for name in chunk_names:
        try:
            s = mesh_summary(name, all_meshes[name])
            summaries.append(s)
        except Exception as e:
            summaries.append({"name": name, "error": str(e)})

    total_chunks = math.ceil(total / chunk_size)
    is_last      = (chunk_idx + 1) >= total_chunks

    return jsonify({
        "status":       "ok",
        "chunk_index":  chunk_idx,
        "chunk_size":   chunk_size,
        "total_meshes": total,
        "total_chunks": total_chunks,
        "is_last":      is_last,
        "summaries":    summaries
    })


def segment():
    """
    Recibe GLB con meshes sin nombre o todo en una sola pieza.
    Intenta segmentarlo en componentes usando:
    1. Split por connected components (piezas no conectadas físicamente)
    2. Ángulo de diedro entre caras adyacentes (fronteras entre planchas)
    3. Clustering por orientación de normales
    """
    data = request.get_json(silent=True) or {}
    glb_url = data.get("glb_url")
    angle_threshold_deg = data.get("angle_threshold_deg", 20)  # grados
    angle_threshold = math.radians(angle_threshold_deg)

    if not glb_url:
        return jsonify({"error": "glb_url requerida"}), 400

    # Descargar GLB
    try:
        req = urllib.request.Request(glb_url, headers={"User-Agent": "Mozilla/5.0 (compatible; ENGRAVIS/1.0)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            glb_bytes = resp.read()
    except Exception as e:
        return jsonify({"error": f"No se pudo descargar: {str(e)}"}), 400

    # Cargar
    try:
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(glb_bytes)
            tmp_path = tmp.name
        scene = trimesh.load(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Error al cargar GLB: {str(e)}"}), 422

    # Extraer meshes
    if isinstance(scene, trimesh.Scene):
        raw_meshes = {n: g for n, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh)}
    elif isinstance(scene, trimesh.Trimesh):
        raw_meshes = {"Mesh_000": scene}
    else:
        return jsonify({"error": "Formato no soportado"}), 422

    segments = []
    seg_id = 0
    method_used = "original"

    for mesh_name, mesh in raw_meshes.items():
        # ── Paso 1: split por connected components ──────────────
        try:
            parts = mesh.split(only_watertight=False)
        except:
            parts = [mesh]

        # Si el split da una sola parte grande, intentar por ángulo de diedro
        if len(parts) == 1 and len(mesh.faces) > 500:
            method_used = "dihedral_angle"
            try:
                # Ángulo entre caras adyacentes
                adj_angles = mesh.face_adjacency_angles  # array de ángulos en radianes
                adj_faces  = mesh.face_adjacency         # pares de caras adyacentes

                # Marcar bordes agudos como fronteras
                sharp = adj_angles > angle_threshold

                # BFS para agrupar caras por conectividad sin cruzar bordes agudos
                face_count = len(mesh.faces)
                visited = np.zeros(face_count, dtype=bool)
                face_groups = []

                # Construir grafo de adyacencia sin bordes agudos
                adjacency = [set() for _ in range(face_count)]
                for i, (f0, f1) in enumerate(adj_faces):
                    if not sharp[i]:
                        adjacency[f0].add(f1)
                        adjacency[f1].add(f0)

                # BFS
                for start in range(face_count):
                    if visited[start]:
                        continue
                    group = []
                    queue = [start]
                    while queue:
                        face = queue.pop(0)
                        if visited[face]:
                            continue
                        visited[face] = True
                        group.append(face)
                        queue.extend([n for n in adjacency[face] if not visited[n]])
                    if len(group) > 5:  # ignorar grupos muy pequeños
                        face_groups.append(group)

                # Crear sub-meshes por grupo de caras
                parts = []
                for group in face_groups:
                    try:
                        sub = mesh.submesh([group], append=True)
                        if sub is not None and hasattr(sub, 'area') and sub.area > 0.001:
                            parts.append(sub)
                    except:
                        pass

                if not parts:
                    parts = [mesh]
            except Exception as e:
                parts = [mesh]

        # Si aún tenemos pocos segments, clustering por orientación
        if len(parts) <= 2 and len(mesh.faces) > 200:
            method_used = "normal_clustering"
            try:
                normals = mesh.face_normals
                # Cuantizar normales en 6 orientaciones principales
                dominant = np.argmax(np.abs(normals), axis=1)  # 0=x, 1=y, 2=z
                sign     = np.sign(normals[np.arange(len(normals)), dominant])
                labels   = dominant * 2 + (sign > 0).astype(int)  # 0-5

                parts = []
                for label in np.unique(labels):
                    face_idx = np.where(labels == label)[0]
                    if len(face_idx) < 10:
                        continue
                    try:
                        sub = mesh.submesh([face_idx.tolist()], append=True)
                        if sub is not None:
                            parts.append(sub)
                    except:
                        pass
                if not parts:
                    parts = [mesh]
            except:
                parts = [mesh]

        # ── Generar summary por parte ────────────────────────────
        for part in parts:
            try:
                s = mesh_summary(f"{mesh_name}_seg{seg_id}", part)
                s["original_mesh"] = mesh_name
                s["segment_id"]    = seg_id

                # Clasificación geométrica rápida
                orient = s.get("orientation", "unknown")
                area   = s.get("area_m2", 0)
                cz     = s.get("centroid", [0,0,0])[2]
                bbox   = s.get("bbox", {})

                if orient == "curved":
                    guess = "casco"
                elif orient == "horizontal" and cz > 0.5:
                    guess = "cubierta"
                elif orient == "horizontal" and cz <= 0:
                    guess = "fondo_quilla"
                elif orient == "vertical" and bbox.get("x",0) > bbox.get("y",0) * 2:
                    guess = "mamparo_longitudinal"
                elif orient == "vertical":
                    guess = "mamparo_transversal"
                elif area < 0.3:
                    guess = "elemento_pequeno"
                else:
                    guess = "estructura"

                s["geometric_guess"] = guess
                segments.append(s)
                seg_id += 1
            except:
                pass

    # Detectar si el modelo estaba sin estructurar
    unstructured = all(
        name.lower().startswith('mesh') or name == 'defaultlayer' or '_' not in name
        for name in raw_meshes.keys()
    )

    return jsonify({
        "status": "ok",
        "original_mesh_count": len(raw_meshes),
        "segment_count": len(segments),
        "method_used": method_used,
        "unstructured": unstructured,
        "angle_threshold_deg": angle_threshold_deg,
        "segments": segments
    })


@app.route('/nest', methods=['POST'])
def nest():
    """
    Compartición de superficies en unidades comerciales estándar.
    Recibe: superficie de un componente + dimensiones de plancha estándar.
    Devuelve: cuántas planchas se necesitan, distribución, desperdicio.

    Formatos de plancha estándar (mm):
    - Acero estructural estándar Ecuador: 2440×1220, 3000×1500, 6000×1500
    - Perfiles: longitudes comerciales 6m, 9m, 12m
    """
    data = request.get_json(silent=True) or {}
    area_m2       = data.get("area_m2", 0)
    bbox          = data.get("bbox", {})          # dimensiones del componente
    component_type = data.get("component_type", "plate")  # plate | profile | pipe
    material_spec  = data.get("material_spec", "A36")
    thickness_mm   = data.get("thickness_mm", 6)

    # Planchas estándar disponibles (largo × ancho en metros)
    STANDARD_PLATES = [
        {"id": "2440x1220", "l": 2.44, "w": 1.22, "area": 2.44*1.22},
        {"id": "3000x1500", "l": 3.00, "w": 1.50, "area": 3.00*1.50},
        {"id": "6000x1500", "l": 6.00, "w": 1.50, "area": 6.00*1.50},
        {"id": "6000x2000", "l": 6.00, "w": 2.00, "area": 6.00*2.00},
        {"id": "12000x2500","l": 12.0, "w": 2.50, "area": 12.0*2.50},
    ]

    # Longitudes estándar de perfiles (metros)
    STANDARD_LENGTHS = [6.0, 9.0, 12.0]

    if component_type == "plate":
        if area_m2 <= 0:
            return jsonify({"error": "area_m2 requerida para planchas"}), 400

        # Factor de desperdicio por curvatura y cortes (empírico naval)
        # Casco curvo: 25-35% desperdicio, planchas planas: 10-15%
        comp_bbox_max = max(bbox.get("x",1), bbox.get("y",1))
        comp_bbox_min = min(bbox.get("x",1), bbox.get("y",1))

        # Estimar curvatura: si el componente es muy curvo
        # la relación área/bbox_area indica pérdida por curvatura
        bbox_area = comp_bbox_max * comp_bbox_min if comp_bbox_max > 0 else area_m2
        curve_ratio = area_m2 / bbox_area if bbox_area > 0 else 1.0

        # Factor de desperdicio
        if curve_ratio > 1.2:        # muy curvo (casco)
            waste_factor = 1.30
        elif curve_ratio > 1.05:     # algo curvo
            waste_factor = 1.18
        else:                        # plano (mamparo, cubierta)
            waste_factor = 1.12

        area_with_waste = area_m2 * waste_factor

        # Encontrar la plancha más eficiente
        results = []
        for plate in STANDARD_PLATES:
            n_plates = math.ceil(area_with_waste / plate["area"])
            total_area_purchased = n_plates * plate["area"]
            actual_waste_pct = (total_area_purchased - area_m2) / total_area_purchased * 100

            # Peso estimado
            density_kg_m3 = 7850
            weight_kg = area_m2 * (thickness_mm / 1000) * density_kg_m3

            results.append({
                "plate_size":          plate["id"],
                "plate_area_m2":       round(plate["area"], 3),
                "quantity":            n_plates,
                "total_area_m2":       round(total_area_purchased, 3),
                "net_area_m2":         round(area_m2, 3),
                "waste_pct":           round(actual_waste_pct, 1),
                "waste_factor_used":   waste_factor,
                "weight_kg":           round(weight_kg, 1),
                "thickness_mm":        thickness_mm,
                "material":            material_spec,
            })

        # Ordenar por menor desperdicio
        results.sort(key=lambda r: r["waste_pct"])

        return jsonify({
            "status":         "ok",
            "component_type": "plate",
            "area_m2":        round(area_m2, 4),
            "recommended":    results[0],
            "alternatives":   results[1:],
        })

    elif component_type == "profile":
        length_m = data.get("length_m", 0)
        if length_m <= 0:
            return jsonify({"error": "length_m requerida para perfiles"}), 400

        results = []
        for std_len in STANDARD_LENGTHS:
            # Piezas por barra
            pieces_per_bar = math.floor(std_len / length_m)
            if pieces_per_bar == 0:
                # El componente es más largo que la barra estándar
                # Se necesitan empalmes
                n_bars = math.ceil(length_m / std_len)
                waste_m = (n_bars * std_len) - length_m
                results.append({
                    "bar_length_m":   std_len,
                    "bars_needed":    n_bars,
                    "splices_needed": n_bars - 1,
                    "waste_m":        round(waste_m, 3),
                    "waste_pct":      round(waste_m / (n_bars * std_len) * 100, 1),
                    "note":           f"Requiere {n_bars-1} empalme(s)"
                })
            else:
                waste_m = std_len - (pieces_per_bar * length_m)
                results.append({
                    "bar_length_m":   std_len,
                    "pieces_per_bar": pieces_per_bar,
                    "bars_needed":    1,
                    "waste_m":        round(waste_m, 3),
                    "waste_pct":      round(waste_m / std_len * 100, 1),
                    "note":           f"{pieces_per_bar} piezas por barra"
                })

        results.sort(key=lambda r: r["waste_pct"])
        return jsonify({
            "status":         "ok",
            "component_type": "profile",
            "length_m":       length_m,
            "recommended":    results[0],
            "alternatives":   results[1:],
        })

    else:
        return jsonify({"error": f"component_type '{component_type}' no soportado. Usar: plate, profile"}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
