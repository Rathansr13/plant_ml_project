/* main.js – PlantGuard AI frontend logic */

const dropZone   = document.getElementById("dropZone");
const dropContent= document.getElementById("dropContent");
const fileInput  = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn   = document.getElementById("clearBtn");
const loader     = document.getElementById("loader");
const results    = document.getElementById("results");

let selectedFile = null;

/* ── Drag & drop ── */
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()=> dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) setFile(f);
});
dropZone.addEventListener("click", ()=> fileInput.click());
fileInput.addEventListener("change", ()=> { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  const url = URL.createObjectURL(f);
  previewImg.src = url;
  previewImg.classList.remove("hidden");
  dropContent.classList.add("hidden");
  analyzeBtn.disabled = false;
  results.classList.add("hidden");
}

clearBtn.addEventListener("click", ()=> {
  selectedFile = null;
  previewImg.src = "";
  previewImg.classList.add("hidden");
  dropContent.classList.remove("hidden");
  analyzeBtn.disabled = true;
  results.classList.add("hidden");
  fileInput.value = "";
});

/* ── Analyze ── */
analyzeBtn.addEventListener("click", async ()=> {
  if (!selectedFile) return;

  loader.classList.remove("hidden");
  results.classList.add("hidden");
  analyzeBtn.disabled = true;

  const fd = new FormData();
  fd.append("image", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) { alert("Error: " + data.error); return; }
    renderResults(data);
  } catch(err) {
    alert("Network error: " + err.message);
  } finally {
    loader.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
});

/* ── Render results ── */
function renderResults(data) {
  const { top3, disease, supplement } = data;
  const best = top3[0];

  // Top prediction
  const name = best.class.replace(/___/g, " → ").replace(/_/g, " ");
  document.getElementById("topPrediction").textContent = name;

  // Confidence bar
  document.getElementById("confBar").style.width = best.confidence + "%";

  // Top 3 list
  const list = document.getElementById("top3List");
  list.innerHTML = top3.map(t =>
    `<div class="top3-item">
       <span>${t.class.replace(/___/g," → ").replace(/_/g," ")}</span>
       <span>${t.confidence}%</span>
     </div>`
  ).join("");

  // Disease info
  document.getElementById("diseaseImg").src  = disease.image_url  || "";
  document.getElementById("diseaseDesc").textContent = disease.description || "";

  // Treatment steps
  const steps = (disease.possible_steps || "").split(/\.\s+|\n/).filter(Boolean);
  document.getElementById("treatmentSteps").innerHTML =
    steps.map(s => `<li>${s.trim()}.</li>`).join("");

  // Supplement
  const suppCard = document.getElementById("supplementCard");
  if (supplement && supplement.supplement_name) {
    document.getElementById("suppImg").src  = supplement.supplement_image || "";
    document.getElementById("suppName").textContent = supplement.supplement_name;
    document.getElementById("suppLink").href = supplement.buy_link || "#";
    suppCard.classList.remove("hidden");
  } else {
    suppCard.classList.add("hidden");
  }

  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth" });
}

/* ── Disease library ── */
async function loadDiseases() {
  const grid = document.getElementById("diseaseGrid");
  try {
    const [dRes, sRes] = await Promise.all([fetch("/diseases"), fetch("/supplements")]);
    const diseases = await dRes.json();
    const supps    = await sRes.json();

    const suppMap = {};
    supps.forEach(s => { suppMap[s.disease_name] = s; });

    window._diseases = diseases;
    renderDiseaseGrid(diseases);
  } catch(e) {
    grid.innerHTML = `<p style="color:#8b949e;text-align:center;">Could not load disease library.</p>`;
  }
}

function renderDiseaseGrid(list) {
  const grid = document.getElementById("diseaseGrid");
  if (!list.length) { grid.innerHTML = `<p class="loading-text">No results found.</p>`; return; }

  grid.innerHTML = list.map(d => {
    const isHealthy = d.disease_name.toLowerCase().includes("healthy");
    const tag = isHealthy ? "Healthy" : "Disease";
    const tagClass = isHealthy ? "healthy" : "";
    return `
      <div class="disease-tile">
        <img src="${d.image_url}" alt="${d.disease_name}"
             onerror="this.src='https://via.placeholder.com/240x130?text=No+Image'"/>
        <div class="disease-tile-body">
          <p class="disease-tile-name">${d.disease_name}</p>
          <span class="disease-tile-tag ${tagClass}">${tag}</span>
        </div>
      </div>`;
  }).join("");
}

document.getElementById("searchBox").addEventListener("input", e => {
  const q = e.target.value.toLowerCase();
  const filtered = (window._diseases || []).filter(d =>
    d.disease_name.toLowerCase().includes(q)
  );
  renderDiseaseGrid(filtered);
});

loadDiseases();
