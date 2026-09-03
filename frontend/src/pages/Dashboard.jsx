import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getDashboard } from "../api";
import { prettyClassName } from "../lib/detections";

export default function Dashboard() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getDashboard()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div style={{ maxWidth: 1160, margin: "0 auto", padding: "32px 24px" }}>
        <div className="card" style={{ padding: 16, color: "var(--text-critical, #c70a24)" }}>
          Couldn't load the dashboard: {error}
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ maxWidth: 1160, margin: "0 auto", padding: "32px 24px" }}>
        <span style={{ fontSize: 13, fontWeight: 550, color: "var(--text-subdued)" }}>Loading…</span>
      </div>
    );
  }

  const stats = [
    { label: "SKUs detected after classification", value: data.stats.classified, note: "crops matched to a SKU embedding" },
    { label: "SKUs detected after verification", value: data.stats.verified, note: "confirmed by OCR" },
    {
      label: "Rejected",
      value: data.stats.rejected_images,
      note: `${data.stats.unknown_in_uploads} unknown crops in Uploads · ${data.stats.unknown_in_rejected} in Rejected`,
    },
  ];

  const colGrid = "minmax(0,1fr) 96px 96px 96px";

  return (
    <div style={{ maxWidth: 1160, margin: "0 auto", padding: "32px 24px 64px", display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <h1 style={{ margin: 0, fontSize: 24, lineHeight: "32px", letterSpacing: "-0.3px", fontWeight: 650 }}>Dashboard</h1>
        <p style={{ margin: 0, fontSize: 13, lineHeight: "20px", fontWeight: 550, color: "var(--text-subdued)" }}>
          All {data.approved_total} images currently in Uploads.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
        {stats.map((stat) => (
          <div key={stat.label} className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>{stat.label}</span>
            <span style={{ fontSize: 30, lineHeight: "40px", letterSpacing: "-0.3px", fontWeight: 650, fontVariantNumeric: "tabular-nums" }}>
              {stat.value}
            </span>
            <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>{stat.note}</span>
          </div>
        ))}
      </div>

      <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600 }}>Products per SKU across uploads</span>
        <div style={{ display: "grid", gridTemplateColumns: colGrid, gap: 8, padding: "6px 0" }}>
          <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600, color: "var(--text-subdued)" }}>SKU</span>
          <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600, color: "var(--text-subdued)", textAlign: "right" }}>Classified</span>
          <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600, color: "var(--text-subdued)", textAlign: "right" }}>Verified</span>
          <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600, color: "var(--text-subdued)", textAlign: "right" }}>Rejected</span>
        </div>
        {data.sku_rows.length === 0 && (
          <span style={{ fontSize: 13, fontWeight: 550, color: "var(--text-subdued)", padding: "8px 0" }}>No SKUs classified yet.</span>
        )}
        {data.sku_rows.map((row) => (
          <div
            key={row.name}
            style={{ display: "grid", gridTemplateColumns: colGrid, gap: 8, padding: "8px 0", borderTop: "1px solid var(--s-border-subdued)", alignItems: "center" }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: row.color, flexShrink: 0 }} />
              <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 550, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {row.name === "Unknown" ? row.name : prettyClassName(row.name)}
              </span>
            </div>
            <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 550, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{row.classified}</span>
            <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 600, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{row.verified}</span>
            <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 550, textAlign: "right", color: "var(--text-subdued)", fontVariantNumeric: "tabular-nums" }}>
              {row.rejected}
            </span>
          </div>
        ))}
      </div>

      <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600 }}>Per image</span>
        {data.image_rows.length === 0 && (
          <span style={{ fontSize: 13, fontWeight: 550, color: "var(--text-subdued)", padding: "8px 0" }}>No images uploaded yet.</span>
        )}
        {data.image_rows.map((row) => (
          <div
            key={row.id}
            style={{ display: "grid", gridTemplateColumns: colGrid, gap: 8, padding: "8px 0", borderTop: "1px solid var(--s-border-subdued)", alignItems: "center" }}
          >
            <button className="link-button" style={{ justifySelf: "start", fontSize: 13, lineHeight: "20px", fontWeight: 550, color: "var(--text-link)" }} onClick={() => navigate(`/uploads/${row.id}`)}>
              {row.name}
            </button>
            <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 550, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{row.classified}</span>
            <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 600, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{row.verified}</span>
            <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 550, textAlign: "right", color: "var(--text-subdued)", fontVariantNumeric: "tabular-nums" }}>
              {row.rejected}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
