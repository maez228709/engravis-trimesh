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


@app.route('/segment', methods=['POST'])
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
        with urllib.request.urlopen(glb_url, timeout=60) as resp:
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
