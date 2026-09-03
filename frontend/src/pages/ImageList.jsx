import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Badge from "../components/Badge";
import ShelfCanvas from "../components/ShelfCanvas";
import { layoutBoxes } from "../lib/detections";
import { listImages } from "../api";

export default function ImageList({ status }) {
  const navigate = useNavigate();
  const [images, setImages] = useState(null);
  const [error, setError] = useState(null);
  const isRejected = status === "rejected";

  useEffect(() => {
    let cancelled = false;
    setImages(null);
    setError(null);
    listImages(status)
      .then((data) => {
        if (!cancelled) setImages(data.images);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [status]);

  const title = isRejected ? "Rejected" : "Uploads";
  const subtitle = isRejected
    ? "Images where unknown crops outnumbered matched Lipton SKUs after OCR verification."
    : "Images that cleared classification and OCR verification.";

  return (
    <div style={{ maxWidth: 1160, margin: "0 auto", padding: "32px 24px 64px", display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <h1 style={{ margin: 0, fontSize: 24, lineHeight: "32px", letterSpacing: "-0.3px", fontWeight: 650 }}>{title}</h1>
          <p style={{ margin: 0, fontSize: 13, lineHeight: "20px", fontWeight: 550, color: "var(--text-subdued)" }}>{subtitle}</p>
        </div>
        {images && <span style={{ fontSize: 12, fontWeight: 550, color: "var(--text-subdued)" }}>{images.length} images</span>}
      </div>

      {error && (
        <div className="card" style={{ padding: 16, color: "var(--text-critical, #c70a24)" }}>
          Couldn't load images: {error}
        </div>
      )}

      {!error && images === null && (
        <span style={{ fontSize: 13, fontWeight: 550, color: "var(--text-subdued)" }}>Loading…</span>
      )}

      {!error && images && images.length === 0 && (
        <span style={{ fontSize: 13, fontWeight: 550, color: "var(--text-subdued)" }}>No images yet.</span>
      )}

      {!error && images && images.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(264px,1fr))", gap: 16, alignItems: "start" }}>
          {images.map((img) => {
            const grey = img.status === "rejected";
            const boxes = layoutBoxes(img.detections, img.image_width, img.image_height);
            return (
              <button
                key={img.id}
                className="image-card"
                onClick={() => navigate(`/${isRejected ? "rejected" : "uploads"}/${img.id}`)}
              >
                <ShelfCanvas
                  boxes={boxes}
                  showLabels={false}
                  imageUrl={img.image_url}
                  imageWidth={img.image_width}
                  imageHeight={img.image_height}
                />
                <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <span
                      style={{
                        fontSize: 13,
                        lineHeight: "20px",
                        fontWeight: 600,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {img.name}
                    </span>
                    <Badge tone={grey ? "critical" : "success"} content={grey ? "Rejected" : "Approved"} />
                  </div>
                  <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>
                    {img.captured} · {img.verified_count} verified · {img.unknown_count} unknown
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
