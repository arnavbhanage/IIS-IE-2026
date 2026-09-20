const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.className = "result";
  result.textContent = "Calculating...";
  const payload = Object.fromEntries(new FormData(form).entries());

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    result.className = `result ${data.placed ? "success" : "error"}`;
    result.innerHTML = `<strong>${data.label}</strong><br>Confidence: ${data.confidence}%<br>${data.explanation}`;
  } catch (error) {
    result.className = "result error";
    result.textContent = error.message;
  }
});