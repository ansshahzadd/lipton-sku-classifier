import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App.jsx";
import { BatchQueueProvider } from "./context/BatchQueueContext";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <BatchQueueProvider>
        <App />
      </BatchQueueProvider>
    </BrowserRouter>
  </StrictMode>
);
