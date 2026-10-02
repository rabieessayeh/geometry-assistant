// Three.js viewer: scene management, camera, picking and the overlays drawn by view actions.
//
// Conventions shared with the back-end: millimetres, Z up, and triangle i of the
// geometry is face i of the tools (the mesh is always loaded from GET /models/{id}/mesh).
//
// Rendering choices:
// - Render on demand. There is no animation loop: a frame is drawn only when the
//   camera, the model, an overlay or the canvas size changes. OrbitControls damping
//   is left off because it needs a continuous loop.
// - Picking uses a bounding volume hierarchy (three-mesh-bvh), so a click does not
//   test every triangle. The BVH is built in "indirect" mode, which leaves the
//   geometry untouched and keeps the face order.
// - Highlights recolour vertices of the one model geometry instead of adding meshes.
// - Everything created for a model or an overlay is disposed when it is replaced.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from "three-mesh-bvh";

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

// Direction from the target to the camera for each standard view. Z stays the
// "up" vector, so the top view is tilted by a hair to keep it well defined.
const VIEW_DIRECTIONS = {
  top: [0, -0.001, 1],
  front: [0, -1, 0],
  side: [1, 0, 0],
  iso: [1, -1, 0.8],
};

const EDGE_ANGLE_DEG = 25; // edges sharper than this are outlined
const MAX_FACES_WITH_EDGES = 150_000; // outlining is skipped on larger meshes
const CLICK_TOLERANCE_PX = 4; // a pointer that moved more than this was a drag
const CAMERA_MOVE_MS = 600;
const FIT_MARGIN = 1.25; // room left around the model by the standard views
const GHOST_OPACITY = 0.3; // highlighted faces seen through the model

function cssColor(name) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return new THREE.Color(value);
}

/** Round a length up to 1, 2 or 5 times a power of ten. */
function niceStep(value) {
  const power = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 5, 10].find((s) => s * power >= value);
  return step * power;
}

function label(text, className, position) {
  const element = document.createElement("div");
  element.className = `label ${className}`;
  element.textContent = text;
  const object = new CSS2DObject(element);
  object.position.copy(position);
  return object;
}

/** Free the GPU resources of an object tree, and the DOM elements of its labels. */
function disposeTree(root) {
  root.traverse((object) => {
    if (object.isCSS2DObject) object.element.remove();
    if (object.geometry && !object.geometry.userData.shared) {
      object.geometry.disposeBoundsTree?.();
      object.geometry.dispose();
    }
    if (object.material && !object.material.userData.shared) object.material.dispose();
  });
}

function shared(resource) {
  resource.userData.shared = true;
  return resource;
}

export class Viewer {
  /**
   * @param {HTMLElement} container element the canvas fills
   * @param {{onPick: (point: number[]) => void, onViewChange: (view: string) => void}} callbacks
   */
  constructor(container, { onPick, onViewChange }) {
    this.container = container;
    this.onPick = onPick;
    this.onViewChange = onViewChange;
    this.colors = {
      base: cssColor("--mesh"),
      edge: cssColor("--mesh-edge"),
      overhang: cssColor("--overhang"),
      normal: cssColor("--normal"),
      slice: cssColor("--slice"),
      dimension: cssColor("--dimension"),
      point: cssColor("--accent"),
      box: cssColor("--muted"),
    };

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.append(this.renderer.domElement);
    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.className = "labels";
    container.append(this.labelRenderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = cssColor("--viewport");
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 1000);
    this.camera.up.set(0, 0, 1);

    const sky = new THREE.HemisphereLight(0xffffff, 0x9aa5b1, 2.0);
    sky.position.set(0, 0, 1);
    // A light that follows the camera keeps every side of the part readable.
    this.headlight = new THREE.DirectionalLight(0xffffff, 1.4);
    this.scene.add(sky, this.headlight, this.headlight.target);

    this.model = null; // THREE.Mesh of the loaded model
    this.radius = 1; // radius of its bounding sphere, used to size everything else
    this.helpers = new THREE.Group(); // grid and axes
    this.markers = new THREE.Group(); // picked points
    this.overlays = new THREE.Group(); // drawn by view actions
    this.scene.add(this.helpers, this.markers, this.overlays);
    this.gridCell = 10;

    this.modelMaterial = shared(
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.75,
        metalness: 0.05,
        side: THREE.DoubleSide, // meshes with flipped faces stay visible
        polygonOffset: true, // lets the edge lines win the depth test
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      }),
    );
    this.edgeMaterial = shared(new THREE.LineBasicMaterial({ color: this.colors.edge }));
    this.markerGeometry = shared(new THREE.SphereGeometry(1, 16, 12));
    this.markerMaterial = shared(
      new THREE.MeshBasicMaterial({ color: this.colors.point, depthTest: false }),
    );

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.userIsMoving = false;
    this.controls.addEventListener("start", () => {
      this.userIsMoving = true;
      this.#stopAnimation();
    });
    this.controls.addEventListener("end", () => (this.userIsMoving = false));
    this.controls.addEventListener("change", () => {
      if (this.userIsMoving) this.#setViewName("custom");
      this.requestRender();
    });

    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = true; // BVH shortcut: stop at the nearest triangle
    this.#listenForClicks();

    this.framePending = false;
    this.animation = null;
    new ResizeObserver(() => this.#resize()).observe(container);
    this.#resize();
  }

  // ------------------------------------------------------------ rendering

  /** Draw one frame at the next animation tick, however many times this is called. */
  requestRender() {
    if (this.framePending) return;
    this.framePending = true;
    requestAnimationFrame(() => {
      this.framePending = false;
      this.headlight.position.copy(this.camera.position);
      this.headlight.target.position.copy(this.controls.target);
      this.renderer.render(this.scene, this.camera);
      this.labelRenderer.render(this.scene, this.camera);
    });
  }

  #resize() {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.labelRenderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  // ---------------------------------------------------------------- model

  /** Replace the displayed model by the binary STL in `buffer`. */
  setModel(buffer) {
    this.clearOverlays();
    this.setPoints([]);
    if (this.model) {
      this.scene.remove(this.model);
      disposeTree(this.model);
    }
    this.#clear(this.helpers);

    const geometry = new STLLoader().parse(buffer);
    geometry.computeBoundsTree({ indirect: true });
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const vertexCount = geometry.attributes.position.count;
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3));

    this.model = new THREE.Mesh(geometry, this.modelMaterial);
    if (vertexCount / 3 <= MAX_FACES_WITH_EDGES) {
      const edges = new THREE.EdgesGeometry(geometry, EDGE_ANGLE_DEG);
      this.model.add(new THREE.LineSegments(edges, this.edgeMaterial));
    }
    this.scene.add(this.model);
    this.#paintBase();

    this.radius = geometry.boundingSphere.radius;
    this.camera.near = this.radius / 100;
    this.camera.far = this.radius * 100;
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = this.radius * 0.05;
    this.controls.maxDistance = this.radius * 20;
    this.#buildHelpers(geometry.boundingBox);
    this.setView("iso", null, { animate: false });
  }

  /** Grid on the plane the model rests on, and axes at the origin. */
  #buildHelpers(box) {
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const extent = Math.max(size.x, size.y, size.z);
    this.gridCell = niceStep(extent / 10);
    const cells = 2 * Math.ceil(extent / this.gridCell) + 2;
    const grid = new THREE.GridHelper(cells * this.gridCell, cells, 0xb6bec8, 0xd3d9e0);
    grid.rotation.x = Math.PI / 2; // GridHelper lies in XZ; the build plate is XY
    grid.position.set(
      Math.round(center.x / this.gridCell) * this.gridCell,
      Math.round(center.y / this.gridCell) * this.gridCell,
      box.min.z - extent * 1e-3, // a hair below the part, to avoid z-fighting with its base
    );
    // The origin triad is drawn over the model: it often lies on one of its edges.
    const axes = new THREE.AxesHelper(extent * 0.75);
    axes.material.depthTest = false;
    axes.renderOrder = 1;
    this.helpers.add(grid, axes);
  }

  #paintBase() {
    const colors = this.model.geometry.attributes.color;
    const { r, g, b } = this.colors.base;
    for (let i = 0; i < colors.count; i++) colors.setXYZ(i, r, g, b);
    colors.needsUpdate = true;
  }

  #clear(group) {
    disposeTree(group);
    group.clear();
  }

  // --------------------------------------------------------------- camera

  /**
   * Move the camera to a standard view of the model.
   * @param {"top"|"front"|"side"|"iso"} name
   * @param {number[]|null} target point to look at; the model centre by default
   */
  setView(name, target = null, { animate = true } = {}) {
    if (!this.model || !VIEW_DIRECTIONS[name]) return;
    const sphere = this.model.geometry.boundingSphere;
    const lookAt = target ? new THREE.Vector3(...target) : sphere.center.clone();
    // Distance at which the bounding sphere fits the narrower of the two fields of view.
    const vertical = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const horizontal = Math.atan(Math.tan(vertical) * this.camera.aspect);
    const distance = (sphere.radius / Math.sin(Math.min(vertical, horizontal))) * FIT_MARGIN;
    const direction = new THREE.Vector3(...VIEW_DIRECTIONS[name]).normalize();
    const position = lookAt.clone().addScaledVector(direction, distance);

    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.#moveCamera(position, lookAt, animate && !still ? CAMERA_MOVE_MS : 0);
    this.#setViewName(name);
  }

  /** Swing the camera around the target instead of cutting through the model. */
  #moveCamera(position, target, duration) {
    this.#stopAnimation();
    if (!duration) {
      this.camera.position.copy(position);
      this.controls.target.copy(target);
      this.controls.update(); // re-aims the camera and triggers a render
      this.requestRender();
      return;
    }
    const fromTarget = this.controls.target.clone();
    const fromOffset = this.camera.position.clone().sub(fromTarget);
    const toOffset = position.clone().sub(target);
    const fromDirection = fromOffset.clone().normalize();
    const swing = new THREE.Quaternion().setFromUnitVectors(
      fromDirection,
      toOffset.clone().normalize(),
    );
    const start = performance.now();

    const step = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const k = t * t * (3 - 2 * t); // ease in and out
      const rotation = new THREE.Quaternion().slerp(swing, k);
      const length = THREE.MathUtils.lerp(fromOffset.length(), toOffset.length(), k);
      this.controls.target.lerpVectors(fromTarget, target, k);
      this.camera.position
        .copy(fromDirection)
        .applyQuaternion(rotation)
        .multiplyScalar(length)
        .add(this.controls.target);
      this.controls.update();
      this.requestRender();
      this.animation = t < 1 ? requestAnimationFrame(step) : null;
    };
    step(start);
  }

  #stopAnimation() {
    if (this.animation) cancelAnimationFrame(this.animation);
    this.animation = null;
  }

  #setViewName(name) {
    if (this.viewName === name) return;
    this.viewName = name;
    this.onViewChange(name);
  }

  // -------------------------------------------------------------- picking

  #listenForClicks() {
    const canvas = this.renderer.domElement;
    let down = null;
    canvas.addEventListener("pointerdown", (event) => {
      down = event.button === 0 ? { x: event.clientX, y: event.clientY } : null;
    });
    canvas.addEventListener("pointerup", (event) => {
      if (!down || !this.model) return;
      const moved = Math.hypot(event.clientX - down.x, event.clientY - down.y);
      down = null;
      if (moved > CLICK_TOLERANCE_PX) return; // it was an orbit, not a click
      const rect = canvas.getBoundingClientRect();
      const pointer = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(pointer, this.camera);
      const [hit] = this.raycaster.intersectObject(this.model, false);
      if (hit) this.onPick(hit.point.toArray());
    });
  }

  /** Show the picked points as numbered markers. */
  setPoints(points) {
    this.#clear(this.markers);
    points.forEach((point, i) => {
      const position = new THREE.Vector3(...point);
      this.markers.add(this.#marker(position), label(`P${i + 1}`, "point", position));
    });
    this.requestRender();
  }

  #marker(position, material = this.markerMaterial) {
    const marker = new THREE.Mesh(this.markerGeometry, material);
    marker.position.copy(position);
    marker.scale.setScalar(this.radius * 0.012);
    marker.renderOrder = 3;
    return marker;
  }

  // ------------------------------------------------------------- overlays

  /** Remove everything drawn by view actions. */
  clearOverlays() {
    this.#clear(this.overlays);
    if (this.model) this.#paintBase();
    this.requestRender();
  }

  /**
   * Colour the given faces; `kind` is "overhang" or "normal".
   *
   * The faces are recoloured on the model itself. A translucent copy is also
   * drawn over the model, so faces hidden from the current viewpoint (the
   * underside of an overhang, for instance) still show through as a ghost.
   */
  highlightFaces(faces, kind) {
    const { position, color: colors } = this.model.geometry.attributes;
    const color = this.colors[kind] ?? this.colors.normal;
    const faceCount = position.count / 3;
    const visible = faces.filter((face) => face < faceCount);
    const ghost = new Float32Array(visible.length * 9);
    visible.forEach((face, i) => {
      for (let corner = 0; corner < 3; corner++) {
        colors.setXYZ(face * 3 + corner, color.r, color.g, color.b);
      }
      // Non-indexed geometry: the nine floats of triangle `face` are contiguous.
      ghost.set(position.array.subarray(face * 9, face * 9 + 9), i * 9);
    });
    colors.needsUpdate = true;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(ghost, 3));
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: GHOST_OPACITY,
      side: THREE.DoubleSide,
      depthTest: false,
    });
    const overlay = new THREE.Mesh(geometry, material);
    overlay.renderOrder = 1;
    this.overlays.add(overlay);
    this.requestRender();
  }

  /** Draw the contours of a cross-section and the cutting plane. */
  showSlice({ axis, value_mm: value, polylines }) {
    const positions = [];
    for (const polyline of polylines) {
      for (let i = 0; i + 1 < polyline.length; i++) positions.push(...polyline[i], ...polyline[i + 1]);
    }
    const contour = new THREE.BufferGeometry();
    contour.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const lines = new THREE.LineSegments(
      contour,
      // Drawn over the model, so the contour stays visible from any side.
      new THREE.LineBasicMaterial({ color: this.colors.slice, depthTest: false }),
    );
    lines.renderOrder = 2;

    // Translucent quad showing where the plane cuts, slightly larger than the model.
    const box = this.model.geometry.boundingBox.clone().expandByScalar(this.radius * 0.08);
    const a = "xyz".indexOf(axis);
    const [u, v] = [0, 1, 2].filter((i) => i !== a);
    const lo = box.min.toArray();
    const hi = box.max.toArray();
    const corners = [
      [lo[u], lo[v]],
      [hi[u], lo[v]],
      [hi[u], hi[v]],
      [lo[u], hi[v]],
    ].map(([cu, cv]) => {
      const corner = [0, 0, 0];
      corner[a] = value;
      corner[u] = cu;
      corner[v] = cv;
      return corner;
    });
    const quad = new THREE.BufferGeometry();
    quad.setAttribute("position", new THREE.Float32BufferAttribute(corners.flat(), 3));
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    const plane = new THREE.Mesh(
      quad,
      new THREE.MeshBasicMaterial({
        color: this.colors.slice,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );

    const text = label(`${axis} = ${value} mm`, "slice", new THREE.Vector3(...corners[2]));
    this.overlays.add(plane, lines, text);
    this.requestRender();
  }

  /** Draw a dimension line between two points, with its length. */
  showDimension({ point_a: a, point_b: b, distance_mm: distance }) {
    const start = new THREE.Vector3(...a);
    const end = new THREE.Vector3(...b);
    const material = new THREE.MeshBasicMaterial({ color: this.colors.dimension, depthTest: false });
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([start, end]),
      new THREE.LineBasicMaterial({ color: this.colors.dimension, depthTest: false }),
    );
    line.renderOrder = 2;
    const middle = start.clone().lerp(end, 0.5);
    this.overlays.add(
      line,
      this.#marker(start, material),
      this.#marker(end, material),
      label(`${distance} mm`, "dimension", middle),
    );
    this.requestRender();
  }

  /** Outline the bounding box and label it with its size. */
  showBoundingBox({ min, max }) {
    const box = new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
    const size = box.getSize(new THREE.Vector3()).toArray();
    const text = `${size.map((s) => +s.toFixed(3)).join(" × ")} mm`;
    this.overlays.add(new THREE.Box3Helper(box, this.colors.box), label(text, "box", box.max));
    this.requestRender();
  }
}
