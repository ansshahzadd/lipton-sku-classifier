import { createContext, useCallback, useContext, useState } from "react";
import { uploadImage, getUploadProgress } from "../api";

// Drives the batch-progress list on Home while files upload to the real
// backend (detect -> embed -> gate -> classify -> OCR runs synchronously
// per file there, so there's one "Processing" stage, not simulated
// sub-stages). Lives above the router so progress survives switching tabs.
// Uploaded images land in Uploads/Rejected/Dashboard via the backend's
// database. Those pages don't poll -- they bump `version` (below) every
// time an upload finishes and refetch in the background off of that,
// without blanking out the data they're already showing.
const BatchQueueContext = createContext(null);

// Real progress: each upload carries a client-generated progress_id that
// the backend reports actual pipeline milestones against (model load,
// detection done, each crop classified -- see pipeline.process_shelf_image's
// on_progress calls). This polls that instead of running a simulated timer.
const POLL_MS = 300;

function makeProgressId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `p-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function BatchQueueProvider({ children }) {
  const [queue, setQueue] = useState([]);
  const [version, setVersion] = useState(0);

  const queueFiles = useCallback((files) => {
    if (!files || !files.length) return;
    const added = Array.from(files).map((f, k) => ({
      key: Date.now() + "-" + k,
      file: f,
      name: f.name,
      stage: "Queued…",
      status: "Processing",
      tone: "info",
      progress: 0,
    }));
    setQueue((prev) => added.concat(prev).slice(0, 8));

    for (const row of added) {
      const progressId = makeProgressId();
      const poll = setInterval(() => {
        getUploadProgress(progressId)
          .then(({ percent, stage }) => {
            if (percent == null) return;
            setQueue((prev) =>
              prev.map((r) => (r.key === row.key ? { ...r, progress: percent, stage: stage || r.stage } : r))
            );
          })
          .catch(() => {
            // Transient poll failure -- next tick retries, nothing to show for it.
          });
      }, POLL_MS);

      uploadImage(row.file, progressId)
        .then((record) => {
          clearInterval(poll);
          const label = record.status === "approved" ? "OCR verified · labelled" : "OCR verified · unknown exceeds Lipton";
          setQueue((prev) =>
            prev.map((r) =>
              r.key === row.key
                ? {
                    ...r,
                    stage: label,
                    status: record.status === "approved" ? "Complete" : "Rejected",
                    tone: record.status === "approved" ? "success" : "critical",
                    progress: 100,
                  }
                : r
            )
          );
          setVersion((v) => v + 1);
        })
        .catch((e) => {
          clearInterval(poll);
          setQueue((prev) =>
            prev.map((r) => (r.key === row.key ? { ...r, stage: e.message, status: "Error", tone: "critical", progress: 0 } : r))
          );
        });
    }
  }, []);

  return <BatchQueueContext.Provider value={{ queue, queueFiles, version }}>{children}</BatchQueueContext.Provider>;
}

export function useBatchQueue() {
  const ctx = useContext(BatchQueueContext);
  if (!ctx) throw new Error("useBatchQueue must be used within BatchQueueProvider");
  return ctx;
}
