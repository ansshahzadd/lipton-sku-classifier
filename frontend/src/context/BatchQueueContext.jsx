import { createContext, useCallback, useContext, useState } from "react";
import { uploadImage } from "../api";

// Drives the batch-progress list on Home while files upload to the real
// backend (detect -> embed -> gate -> classify -> OCR runs synchronously
// per file there, so there's one "Processing" stage, not simulated
// sub-stages). Lives above the router so progress survives switching tabs.
// Uploaded images land in Uploads/Rejected/Dashboard via the backend's
// database — those pages fetch fresh on mount, so nothing needs to be
// pushed there from here.
const BatchQueueContext = createContext(null);

export function BatchQueueProvider({ children }) {
  const [queue, setQueue] = useState([]);

  const queueFiles = useCallback((files) => {
    if (!files || !files.length) return;
    const added = Array.from(files).map((f, k) => ({
      key: Date.now() + "-" + k,
      file: f,
      name: f.name,
      stage: "Uploading & processing…",
      status: "Processing",
      tone: "info",
    }));
    setQueue((prev) => added.concat(prev).slice(0, 8));

    for (const row of added) {
      uploadImage(row.file)
        .then((record) => {
          const label = record.status === "approved" ? "OCR verified · labelled" : "OCR verified · unknown exceeds Lipton";
          setQueue((prev) =>
            prev.map((r) =>
              r.key === row.key
                ? { ...r, stage: label, status: record.status === "approved" ? "Complete" : "Rejected", tone: record.status === "approved" ? "success" : "critical" }
                : r
            )
          );
        })
        .catch((e) => {
          setQueue((prev) =>
            prev.map((r) => (r.key === row.key ? { ...r, stage: e.message, status: "Error", tone: "critical" } : r))
          );
        });
    }
  }, []);

  return <BatchQueueContext.Provider value={{ queue, queueFiles }}>{children}</BatchQueueContext.Provider>;
}

export function useBatchQueue() {
  const ctx = useContext(BatchQueueContext);
  if (!ctx) throw new Error("useBatchQueue must be used within BatchQueueProvider");
  return ctx;
}
