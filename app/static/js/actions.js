// Applies the view actions returned by the geometry tools to the viewer.
//
// A view action is plain data: {type, ...}. The back-end decides what to show;
// this module only maps each type to a viewer method and describes it in the legend.

const HANDLERS = {
  highlight_faces(viewer, action) {
    viewer.highlightFaces(action.faces, action.kind);
    const color = action.kind === "overhang" ? "--overhang" : "--normal";
    return { color, text: `${action.faces.length} faces: ${action.label}` };
  },
  slice(viewer, action) {
    viewer.showSlice(action);
    return { color: "--slice", text: `Section at ${action.axis} = ${action.value_mm} mm` };
  },
  dimension(viewer, action) {
    viewer.showDimension(action);
    return { color: "--dimension", text: `Distance: ${action.distance_mm} mm` };
  },
  bounding_box(viewer, action) {
    viewer.showBoundingBox(action);
    return { color: "--muted", text: "Bounding box" };
  },
  camera(viewer, action) {
    viewer.setView(action.view, action.target);
    return null;
  },
};

/** Id of the model the actions refer to, if any. */
export function modelOf(actions) {
  return actions.find((action) => action.model)?.model ?? null;
}

/**
 * Replace the current overlays by those of `actions`.
 * @returns {{color: string, text: string}[]} legend entries, one per overlay drawn
 */
export function applyViewActions(viewer, actions) {
  viewer.clearOverlays();
  const legend = [];
  for (const action of actions) {
    const entry = HANDLERS[action.type]?.(viewer, action);
    if (entry) legend.push(entry);
  }
  return legend;
}
