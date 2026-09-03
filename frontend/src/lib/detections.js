// Turns a backend image record's `detections` (pixel bboxes against the
// real photo) into the percentage-based box styles ShelfCanvas expects.
//
// Box color/tag is driven by the crop's decision, not by which SKU it is —
// that's what makes a busy shelf photo scannable at a glance:
//   EXACT_SKU              -> purple box, tagged with the SKU name
//   UNKNOWN                -> yellow box, tagged "Unknown"
//   REJECT / NON_TARGET    -> dull grey box, no tag at all -- neither a
//                             confirmed SKU nor a genuine "unknown target",
//                             just a crop that didn't clear an earlier gate
//                             (bad crop, or not a BP product), so it isn't
//                             worth a label, but it's still drawn so you can
//                             see what the detector found.

export const GREY = "rgb(138,138,138)";

export const DECISION_COLORS = {
  EXACT_SKU: "#7c3aed",
  UNKNOWN: "#eab308",
  REJECT: GREY,
  NON_TARGET: GREY,
};

export function prettyClassName(name) {
  if (!name) return "Unknown";
  return name
    .split("-")
    .map((w) => (w.length ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

export function layoutBoxes(detections, imageWidth, imageHeight) {
  if (!imageWidth || !imageHeight) return [];
  return detections.map((d) => {
    const [x1, y1, x2, y2] = d.bbox;
    const color = DECISION_COLORS[d.decision] || DECISION_COLORS.UNKNOWN;
    const geom = {
      left: (x1 / imageWidth) * 100 + "%",
      top: (y1 / imageHeight) * 100 + "%",
      width: ((x2 - x1) / imageWidth) * 100 + "%",
      height: ((y2 - y1) / imageHeight) * 100 + "%",
    };
    const showTag = d.decision === "EXACT_SKU" || d.decision === "UNKNOWN";
    const label = d.decision === "EXACT_SKU" ? prettyClassName(d.class_name) : "Unknown";
    return {
      label,
      showTag,
      style: { ...geom, border: `2px solid ${color}`, background: "transparent" },
      thumbStyle: { ...geom, border: `1px solid ${color}`, opacity: 0.9 },
      color,
    };
  });
}
