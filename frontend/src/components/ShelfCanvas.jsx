import { useState } from "react";

// Renders the real shelf photo (sized to its true aspect ratio so the
// percentage-positioned boxes line up exactly) with detection boxes drawn
// on top. Falls back to a neutral striped placeholder when no photo is
// available yet (e.g. still uploading).
//
// In label mode (Preview), boxes/tags render at full opacity until you
// click one -- then that box stays full opacity with its tag expanded to
// the full, untruncated label, and every other box dims out of the way so
// you can check it against the underlying photo without the rest of the
// shelf's clutter. Clicking empty canvas clears the selection.
const STRIPES =
  "linear-gradient(to bottom, transparent 32%, rgba(0,0,0,0.06) 32%, rgba(0,0,0,0.06) 33%, transparent 33%, transparent 65%, rgba(0,0,0,0.06) 65%, rgba(0,0,0,0.06) 66%, transparent 66%)";

const DIM_OPACITY = 0.35;

export default function ShelfCanvas({ boxes, height, aspectRatio, showLabels, imageUrl, imageWidth, imageHeight }) {
  const [selected, setSelected] = useState(null);
  const ratio = imageUrl && imageWidth && imageHeight ? `${imageWidth} / ${imageHeight}` : aspectRatio;

  return (
    <div
      onClick={() => showLabels && setSelected(null)}
      style={{
        position: "relative",
        width: "100%",
        height: ratio ? undefined : height,
        aspectRatio: ratio,
        borderRadius: aspectRatio ? 8 : 0,
        background: "var(--s-box-subdued)",
        backgroundImage: imageUrl ? undefined : STRIPES,
        borderBottom: aspectRatio ? undefined : "1px solid var(--s-border-subdued)",
        boxShadow: aspectRatio ? "inset 0 0 0 1px rgba(0,0,0,0.10)" : undefined,
        overflow: "hidden",
      }}
    >
      {imageUrl && (
        <img
          src={imageUrl}
          alt=""
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
        />
      )}
      {boxes.map((box, i) => {
        const isSelected = showLabels && i === selected;
        return (
          <div
            key={i}
            onClick={(e) => {
              if (!showLabels) return;
              e.stopPropagation();
              setSelected(isSelected ? null : i);
            }}
            style={{
              position: "absolute",
              boxSizing: "border-box",
              borderRadius: showLabels ? 4 : 2,
              cursor: showLabels ? "pointer" : undefined,
              opacity: showLabels && selected !== null && !isSelected ? DIM_OPACITY : 1,
              zIndex: isSelected ? 2 : 1,
              transition: "opacity 120ms ease",
              ...(showLabels ? box.style : box.thumbStyle),
            }}
          >
            {showLabels && box.showTag !== false && (
              <span
                style={{
                  position: "absolute",
                  top: -9,
                  left: -2,
                  background: box.color,
                  color: "#fff",
                  font: "600 10px/14px var(--font-sans)",
                  padding: "0 4px",
                  borderRadius: 2,
                  whiteSpace: "nowrap",
                  maxWidth: isSelected ? "none" : "calc(100% + 4px)",
                  overflow: isSelected ? "visible" : "hidden",
                  textOverflow: isSelected ? "clip" : "ellipsis",
                }}
              >
                {box.label}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
