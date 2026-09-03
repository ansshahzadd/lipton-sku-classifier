import { useEffect, useRef, useState } from "react";
import Button from "../components/Button";
import Badge from "../components/Badge";
import UploadIcon from "../components/icons/UploadIcon";
import { useBatchQueue } from "../context/BatchQueueContext";

const STEPS = [
  { n: "01", title: "Cropper", body: "Generates one crop per product found on the shelf." },
  { n: "02", title: "Classifier", body: "Matches every crop against the SKU embedding index." },
  { n: "03", title: "OCR", body: "Reads the pack text to confirm or reject the predicted SKU." },
];

export default function Home() {
  const [drag, setDrag] = useState(false);
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const { queue, queueFiles } = useBatchQueue();

  useEffect(() => {
    // webkitdirectory has no JSX prop and no React-recognized attribute name.
    if (folderInputRef.current) folderInputRef.current.webkitdirectory = true;
  }, []);

  return (
    <div style={{ maxWidth: 920, margin: "0 auto", padding: "48px 24px 64px", display: "flex", flexDirection: "column", gap: 32 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <h1 style={{ margin: 0, fontSize: 30, lineHeight: "40px", letterSpacing: "-0.3px", fontWeight: 650 }}>Upload shelf images</h1>
        <p style={{ margin: 0, fontSize: 14, lineHeight: "20px", color: "var(--text-subdued)", maxWidth: "56ch", textWrap: "pretty" }}>
          Every image runs through the same pipeline: the cropper generates a crop per product, the classifier matches each crop
          against SKU embeddings, and OCR verifies the label before the crop is tagged with its SKU.
        </p>
      </div>

      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!drag) setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          queueFiles(e.dataTransfer?.files);
        }}
        style={{
          border: `1px dashed ${drag ? "rgb(0,91,211)" : "var(--s-border-strong)"}`,
          background: drag ? "rgba(0,91,211,0.04)" : "var(--s-box-base)",
          borderRadius: 12,
          padding: "48px 24px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          cursor: "pointer",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: "var(--radius-full)",
              background: "var(--s-box-subdued)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <UploadIcon />
          </div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>Drag and drop images here</div>
          <div style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>
            JPG or PNG. One image or a whole batch.
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 16 }}>
          <Button
            variant="primary"
            label="Select files"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          />
          <Button
            variant="tertiary"
            label="Select folder"
            onClick={(e) => {
              e.stopPropagation();
              folderInputRef.current?.click();
            }}
          />
        </div>
        <input
          type="file"
          accept="image/*"
          multiple
          ref={fileInputRef}
          onChange={(e) => queueFiles(e.target.files)}
          onClick={(e) => e.stopPropagation()}
          style={{ display: "none" }}
        />
        <input
          type="file"
          accept="image/*"
          multiple
          ref={folderInputRef}
          onChange={(e) => queueFiles(e.target.files)}
          onClick={(e) => e.stopPropagation()}
          style={{ display: "none" }}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
        {STEPS.map((step) => (
          <div key={step.n} className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600, color: "var(--text-subdued)" }}>{step.n}</span>
            <span style={{ fontSize: 13, lineHeight: "20px", fontWeight: 600 }}>{step.title}</span>
            <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>{step.body}</span>
          </div>
        ))}
      </div>

      {queue.length > 0 && (
        <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600 }}>This batch</span>
          {queue.map((row) => (
            <div
              key={row.key}
              style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderTop: "1px solid var(--s-border-subdued)" }}
            >
              <div style={{ width: 32, height: 32, borderRadius: 6, background: "var(--s-box-subdued)", flexShrink: 0 }} />
              <div style={{ flexGrow: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, lineHeight: "20px", fontWeight: 550, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {row.name}
                </div>
                <div style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>{row.stage}</div>
              </div>
              <Badge tone={row.tone} content={row.status} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
