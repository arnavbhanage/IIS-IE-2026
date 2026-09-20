const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

const CATEGORY_COLORS = { A: "#16a34a", B: "#d97706", C: "#dc2626" };

function renderBars(containerId, obj, colorMap) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  Object.entries(obj).forEach(([label, value]) => {
    const pct = (value * 100).toFixed(1);
    const row = document.createElement("div");
    row.className = "bar-row";

    const color = colorMap && colorMap[label] ? colorMap[label] : "#2563eb";

    row.innerHTML = `
      <div class="bar-label">${label}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct}%; background:${color}"></div>
      </div>
      <div class="bar-value">${pct}%</div>
    `;
    container.appendChild(row);
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    payload[key] = Number(value);
  });

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    document.getElementById("pred-category").textContent = data.prediction;
    document.getElementById("pred-category").style.color = CATEGORY_COLORS[data.prediction] || "#111";

    renderBars("prob-bars", data.probabilities, CATEGORY_COLORS);

    const pathList = document.getElementById("decision-path");
    pathList.innerHTML = "";
    data.decision_path.forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      pathList.appendChild(li);
    });
    const finalLi = document.createElement("li");
    finalLi.textContent = `=> Final prediction: ${data.prediction}`;
    finalLi.style.fontWeight = "700";
    pathList.appendChild(finalLi);

    renderBars("local-contrib", data.local_feature_contributions);
    renderBars("global-importance", data.global_feature_importance);

    resultBox.classList.remove("hidden");
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  }
});
