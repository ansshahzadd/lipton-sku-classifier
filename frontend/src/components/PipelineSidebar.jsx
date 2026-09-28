import { prettyClassName } from "../lib/detections";

// Step-by-step breakdown of the detect -> gate -> classify -> verify
// cascade for whichever product box is currently selected on the Preview
// page. Every stage always renders, in cascade order, even ones the crop
// never reached -- that's what makes "where did it stop" legible at a
// glance instead of just showing the final decision.
const STAGES = [
  { key: "reject_gate", label: "Reject Gate" },
  { key: "bp_gate", label: "BP Target Gate" },
  { key: "sku_head", label: "SKU Classifier" },
  { key: "ocr", label: "OCR Verification" },
];

const OUTCOME_LABEL = {
  EXACT_SKU: "Classified",
  UNKNOWN: "Unknown",
  REJECT: "Rejected — invalid crop",
  NON_TARGET: "Rejected — not a Lipton product",
};

const OUTCOME_COLOR = {
  EXACT_SKU: "#7c3aed",
  UNKNOWN: "#eab308",
  REJECT: "rgb(138,138,138)",
  NON_TARGET: "rgb(138,138,138)",
};

const STATE_COLOR = {
  pass: "#16a34a",
  fail: "#dc2626",
  skip: "var(--s-border-subdued)",
};

// Whole-image (not per-crop) duration summary, same 6 guide stages the
// terminal timing report groups into -- shown in this sidebar's idle state
// (nothing selected) so "how long did this image take" is visible without
// digging through server logs.
const TIMING_STAGES = [
  { key: "detect", label: "Detect" },
  { key: "embed", label: "Embed" },
  { key: "gate", label: "Gate" },
  { key: "classify", label: "Classify" },
  { key: "verify", label: "Verify (OCR)" },
  { key: "decide", label: "Decide" },
  { key: "db_save", label: "Database save" },
];

function fmt(n) {
  return typeof n === "number" ? n.toFixed(3) : "—";
}

function formatDuration(seconds) {
  if (typeof seconds !== "number") return "—";
  if (seconds >= 60) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  }
  return `${seconds.toFixed(1)}s`;
}

function describeStep(stage, step) {
  if (!step) {
    return { state: "skip", title: "Not reached", detail: "The pipeline stopped at an earlier stage." };
  }
  if (stage === "reject_gate") {
    if (!step.active) {
      return { state: "skip", title: "Inactive", detail: "No reject gate installed — every crop passes through." };
    }
    return {
      state: step.passed ? "pass" : "fail",
      title: step.passed ? "Passed" : "Rejected",
      detail: `score ${fmt(step.score)} vs threshold ${fmt(step.threshold)}`,
    };
  }
  if (stage === "bp_gate") {
    if (!step.active) {
      return { state: "skip", title: "Inactive", detail: "No BP target gate installed — every crop passes through." };
    }
    return {
      state: step.passed ? "pass" : "fail",
      title: step.passed ? "Passed" : "Not a Lipton product",
      detail: `score ${fmt(step.score)} vs threshold ${fmt(step.threshold)}`,
    };
  }
  if (stage === "sku_head") {
    const modeLabel = step.mode === "sku_head" ? "SKU classifier" : "cosine match (fallback)";
    return {
      state: step.passed ? "pass" : "fail",
      title: step.passed ? `Matched ${prettyClassName(step.top_class)}` : "No confident match",
      detail: `${modeLabel} — score ${fmt(step.score)} vs threshold ${fmt(step.threshold)}, margin ${fmt(step.margin)}`,
    };
  }
  if (stage === "ocr") {
    return {
      state: step.passed ? "pass" : "fail",
      title: step.passed ? `Confirmed ${prettyClassName(step.matched_class)}` : "Could not confirm",
      detail: step.passed
        ? "Keyword/size match found in the crop's text."
        : "No keyword or size match found in the crop's text.",
    };
  }
  return { state: "skip", title: "—", detail: "" };
}

export default function PipelineSidebar({ detection, timings }) {
  if (!detection) {
    return (
      <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600 }}>Pipeline breakdown</span>
        <span style={{ fontSize: 12, lineHeight: "18px", color: "var(--text-subdued)", marginBottom: 8 }}>
          Click a product box to see how it was classified, step by step.
        </span>

        <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600, marginTop: 4 }}>Processing time</span>
        {timings ? (
          <>
            {TIMING_STAGES.map(({ key, label }) => (
              <div
                key={key}
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  justifyContent: "space-between",
                  gap: 12,
                  padding: "6px 0",
                  borderTop: "1px solid var(--s-border-subdued)",
                }}
              >
                <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 550, color: "var(--text-subdued)" }}>{label}</span>
                <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                  {formatDuration(timings[key])}
                </span>
              </div>
            ))}
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                justifyContent: "space-between",
                gap: 12,
                padding: "8px 0 0",
                marginTop: 4,
                borderTop: "1px solid var(--s-border-subdued)",
              }}
            >
              <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 600 }}>Total</span>
              <span style={{ fontSize: 14, lineHeight: "18px", fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
                {formatDuration(timings.total)}
              </span>
            </div>
          </>
        ) : (
          <span style={{ fontSize: 12, lineHeight: "18px", color: "var(--text-subdued)" }}>
            No timing data recorded for this image.
          </span>
        )}
      </div>
    );
  }

  const stepsByStage = Object.fromEntries((detection.steps || []).map((s) => [s.stage, s]));
  const outcomeLabel = OUTCOME_LABEL[detection.decision] || detection.decision || "Unknown";
  const outcomeColor = OUTCOME_COLOR[detection.decision] || OUTCOME_COLOR.UNKNOWN;

  return (
    <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 11, lineHeight: "12px", fontWeight: 600, marginBottom: 4 }}>Pipeline breakdown</span>
      {STAGES.map(({ key, label }) => {
        const step = stepsByStage[key];
        const { state, title, detail } = describeStep(key, step);
        return (
          <div
            key={key}
            style={{ display: "flex", gap: 10, padding: "8px 0", borderTop: "1px solid var(--s-border-subdued)" }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: STATE_COLOR[state],
                flexShrink: 0,
                marginTop: 4,
              }}
            />
            <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
              <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 600 }}>{label}</span>
              <span
                style={{
                  fontSize: 12,
                  lineHeight: "16px",
                  fontWeight: 550,
                  color: state === "fail" ? "#dc2626" : "var(--text-subdued)",
                }}
              >
                {title}
              </span>
              {detail && (
                <span style={{ fontSize: 11, lineHeight: "15px", color: "var(--text-subdued)" }}>{detail}</span>
              )}
            </div>
          </div>
        );
      })}

      <div
        style={{
          marginTop: 8,
          padding: "10px 12px",
          borderRadius: 6,
          background: "var(--s-box-subdued)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div style={{ width: 8, height: 8, borderRadius: 2, background: outcomeColor, flexShrink: 0 }} />
        <span style={{ fontSize: 12, lineHeight: "16px", fontWeight: 600 }}>
          {detection.decision === "EXACT_SKU" ? `Classified as ${prettyClassName(detection.class_name)}` : outcomeLabel}
        </span>
      </div>
    </div>
  );
}
