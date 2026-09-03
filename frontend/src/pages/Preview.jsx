import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Badge from "../components/Badge";
import ShelfCanvas from "../components/ShelfCanvas";
import { DECISION_COLORS, GREY, layoutBoxes, prettyClassName } from "../lib/detections";
import { getImage } from "../api";

export default function Preview({ from }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const [img, setImg] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setImg(null);
    setError(null);
    getImage(id)
      .then((data) => {
        if (!cancelled) setImg(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const backLabel = from === "rejected" ? "Rejected" : "Uploads";

  if (error) {
    return (
      <div style={{ maxWidth: 1160, margin: "0 auto", padding: "24px" }}>
        <button className="back-button" onClick={() => navigate(`/${from}`)}>
          {backLabel}
        </button>
        <div className="card" style={{ padding: 16, marginTop: 16, color: "var(--text-critical, #c70a24)" }}>
          Couldn't load this image: {error}
        </div>
      </div>
    );
  }

  if (!img) return null;

  const grey = img.status === "rejected";
  const boxes = layoutBoxes(img.detections, img.image_width, img.image_height);

  // Per-crop decision tallies, straight from the cascade (guide stage 6):
  // EXACT_SKU/UNKNOWN both passed the classification gate and went to OCR;
  // REJECT (bad crop) and NON_TARGET (not a BP product) never got that far.
  const verifiedByClass = {};
  let unknownCount = 0;
  let gatedOutCount = 0; // REJECT + NON_TARGET -- never reached classification
  for (const d of img.detections) {
    if (d.decision === "EXACT_SKU") {
      verifiedByClass[d.class_name] = (verifiedByClass[d.class_name] || 0) + 1;
    } else if (d.decision === "UNKNOWN") {
      unknownCount += 1;
    } else {
      gatedOutCount += 1;
    }
  }
  const verifiedCount = Object.values(verifiedByClass).reduce((a, n) => a + n, 0);
  const classifiedCount = verifiedCount + unknownCount; // matched a candidate SKU embedding, went to OCR

  const rows = Object.entries(verifiedByClass).map(([className, count]) => ({
    sku: prettyClassName(className),
    count,
    color: DECISION_COLORS.EXACT_SKU,
  }));
  if (unknownCount > 0) rows.push({ sku: "Unknown", count: unknownCount, color: DECISION_COLORS.UNKNOWN });
  if (gatedOutCount > 0) rows.push({ sku: "Rejected", count: gatedOutCount, color: GREY });

  const stats = [
    { label: "SKUs detected after classification", value: classifiedCount },
    { label: "SKUs detected after verification", value: verifiedCount },
    { label: "Rejected", value: gatedOutCount + unknownCount },
  ];

  const legend = [
    { label: "Verified SKU", swatch: DECISION_COLORS.EXACT_SKU },
    { label: "Unknown", swatch: DECISION_COLORS.UNKNOWN },
    { label: "Rejected crop (unusable)", swatch: DECISION_COLORS.REJECT },
  ];

  return (
    <div style={{ maxWidth: 1160, margin: "0 auto", padding: "24px 24px 64px", display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button className="back-button" onClick={() => navigate(`/${from}`)}>
          {backLabel}
        </button>
        <span style={{ fontSize: 12, color: "var(--text-disabled)" }}>/</span>
        <span style={{ fontSize: 12, fontWeight: 550, color: "var(--text-subdued)" }}>{img.name}</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 320px", gap: 16, alignItems: "start" }}>
        <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ fontSize: 14, lineHeight: "20px", fontWeight: 600 }}>{img.name}</span>
              <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>
                {img.captured} · {img.detections.length} crops
              </span>
            </div>
            <Badge tone={grey ? "critical" : "success"} content={grey ? "Rejected · unknown exceeds Lipton" : "Approved"} />
          </div>

          <ShelfCanvas
            boxes={boxes}
            aspectRatio="16 / 10"
            showLabels
            imageUrl={img.image_url}
            imageWidth={img.image_width}
            imageHeight={img.image_height}
          />

          <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            {legend.map((key) => (
              <div key={key.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: 2,
                    border: `2px solid ${key.swatch}`,
                    background: key.dashed ? "rgba(138,138,138,0.10)" : "transparent",
                  }}
                />
                <span style={{ fontSize: 12, fontWeight: 550, color: "var(--text-subdued)" }}>{key.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600 }}>Products in this image</span>
            {rows.map((row) => (
              <div
                key={row.sku}
                style={{ display: "flex", alignItems: "center", gap: 12, padding: "6px 0", borderTop: "1px solid var(--s-border-subdued)" }}
              >
                <div style={{ width: 8, height: 8, borderRadius: 2, background: row.color, flexShrink: 0 }} />
                <span style={{ flexGrow: 1, fontSize: 13, lineHeight: "20px", fontWeight: 550 }}>{row.sku}</span>
                <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>{row.count}</span>
              </div>
            ))}
          </div>

          <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600 }}>Pipeline result</span>
            {stats.map((stat) => (
              <div
                key={stat.label}
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  justifyContent: "space-between",
                  gap: 12,
                  padding: "6px 0",
                  borderTop: "1px solid var(--s-border-subdued)",
                }}
              >
                <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)", maxWidth: "20ch", textWrap: "pretty" }}>
                  {stat.label}
                </span>
                <span style={{ fontSize: 18, lineHeight: "24px", letterSpacing: "-0.2px", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                  {stat.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
