// Inférence côté client : télécharge model.onnx + preprocess.json + config.json
// depuis le dépôt de modèle HF (config.json pointe la révision publiée par la CI),
// applique le même log1p que scikit-learn côté serveur, puis lance ONNX Runtime Web.

let session, preprocess, modelMeta;

async function loadModel() {
  const config = await (await fetch("config.json")).json();
  const base = `https://huggingface.co/${config.model_repo}/resolve/${config.model_revision}`;

  preprocess = await (await fetch(`${base}/preprocess.json`)).json();
  modelMeta = await (await fetch(`${base}/model_meta.json`)).json();
  session = await ort.InferenceSession.create(`${base}/model.onnx`);

  document.getElementById("meta").innerText =
    `Modèle ${modelMeta.model_name} v${modelMeta.model_version} — run ${modelMeta.run_id} — seuil ${modelMeta.decision_threshold}`;
}

function buildFeed(formData) {
  const feed = {};
  for (const col of preprocess.numeric) {
    feed[col] = new ort.Tensor("float32", [parseFloat(formData.get(col) || 0)], [1, 1]);
  }
  for (const col of preprocess.numeric_log1p) {
    const raw = parseFloat(formData.get(col) || 0);
    feed[col] = new ort.Tensor("float32", [Math.log1p(raw)], [1, 1]);
  }
  for (const col of preprocess.categorical) {
    feed[col] = new ort.Tensor("string", [formData.get(col) || "missing"], [1, 1]);
  }
  return feed;
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!session) await loadModel();

  const feed = buildFeed(new FormData(e.target));
  const output = await session.run(feed);
  const proba = output.probabilities.data[1];
  const label = proba >= preprocess.decision_threshold ? ">50K" : "<=50K";

  const el = document.getElementById("result");
  el.hidden = false;
  el.innerHTML = `<strong>${label}</strong> — score ${proba.toFixed(4)} (seuil ${preprocess.decision_threshold})`;
});

loadModel().catch((err) => {
  document.getElementById("meta").innerText = "Modèle indisponible : " + err.message;
});
