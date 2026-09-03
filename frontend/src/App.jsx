import { Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import ImageList from "./pages/ImageList";
import Preview from "./pages/Preview";
import Dashboard from "./pages/Dashboard";

export default function App() {
  return (
    <div style={{ minHeight: "100vh", background: "var(--s-box-base)" }}>
      <Navbar brandName="Lipton SKU Classifier" />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/uploads" element={<ImageList status="approved" />} />
        <Route path="/rejected" element={<ImageList status="rejected" />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/uploads/:id" element={<Preview from="uploads" />} />
        <Route path="/rejected/:id" element={<Preview from="rejected" />} />
      </Routes>
    </div>
  );
}
