// Entry point: wires the model picker, the viewer, the selection panel and the chat.

import { applyViewActions, modelOf } from "./actions.js";
import * as api from "./api.js";
import { Chat } from "./chat.js";
import { Viewer } from "./viewer.js";

const MAX_POINTS = 10; // same limit as the API
const MAX_HISTORY = 10; // previous messages sent with a question
const MAX_TURN_CHARS = 4000;

// Example questions: [label of the button, question sent].
const SUGGESTIONS = [
  ["Volume and bounding box", "What is the volume and bounding box?"],
  ["Support for 3D printing", "Highlight the faces that would need support for 3D printing"],
  ["Slice at z = 10 mm", "Slice the part at z = 10 mm"],
  ["Top view", "Show me the top view"],
  ["Distance between picked points", "Measure the distance between the two points I clicked"],
];

const $ = (id) => document.getElementById(id);

const state = {
  models: [], // summaries from GET /models
  active: null, // id of the model on screen
  points: [], // picked points, [x, y, z] in mm
  view: "iso",
  history: [], // {role, content} turns of the conversation
  busy: false,
};

const chat = new Chat($("messages"));
const viewer = new Viewer($("viewer"), {
  onPick(point) {
    const rounded = point.map((value) => Math.round(value * 1000) / 1000);
    state.points = [...state.points, rounded].slice(-MAX_POINTS);
    renderPoints();
  },
  onViewChange(view) {
    state.view = view;
    $("current-view").textContent = view;
  },
});

// ------------------------------------------------------------------ models

function showStatus(message) {
  $("status").textContent = message ?? "";
  $("status").hidden = !message;
}

async function refreshModels() {
  state.models = await api.listModels();
  const options = state.models.map((model) => {
    const label = model.source === "upload" ? `${model.id} (uploaded)` : model.id;
    return new Option(label, model.id);
  });
  $("model-select").replaceChildren(...options);
  if (state.active) $("model-select").value = state.active;
}

let loadCounter = 0;

async function selectModel(id) {
  const load = ++loadCounter;
  const buffer = await api.fetchMesh(id);
  if (load !== loadCounter) return; // another model was selected meanwhile
  state.active = id;
  state.points = [];
  $("model-select").value = id;
  viewer.setModel(buffer);
  $("grid-cell").textContent = viewer.gridCell;
  renderFacts();
  renderPoints();
  renderLegend([]);
}

function renderFacts() {
  const model = state.models.find((m) => m.id === state.active);
  if (!model) return;
  const facts = {
    Faces: model.faces.toLocaleString("en"),
    Vertices: model.vertices.toLocaleString("en"),
    "Size (mm)": model.bounding_box_mm.size.join(" × "),
    Watertight: model.watertight ? "yes" : "no",
  };
  $("model-facts").replaceChildren(
    ...Object.entries(facts).flatMap(([name, value]) => {
      const term = document.createElement("dt");
      term.textContent = name;
      const description = document.createElement("dd");
      description.textContent = value;
      return [term, description];
    }),
  );
}

// --------------------------------------------------------------- selection

function renderPoints() {
  viewer.setPoints(state.points);
  $("points").replaceChildren(
    ...state.points.map((point, i) => {
      const item = document.createElement("li");
      const name = document.createElement("b");
      name.textContent = `P${i + 1}`;
      item.append(name, point.join(", "));
      return item;
    }),
  );
  $("clear-points").hidden = state.points.length === 0;
  $("points-hint").hidden = state.points.length > 0;
}

function renderLegend(entries) {
  $("legend").hidden = entries.length === 0;
  $("legend").replaceChildren(
    ...entries.map(({ color, text }) => {
      const row = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.background = `var(${color})`;
      row.append(swatch, text);
      return row;
    }),
  );
}

// -------------------------------------------------------------------- chat

/** Draw the view actions of an answer, switching to the model they refer to if needed. */
async function show(actions) {
  const model = modelOf(actions);
  try {
    if (model && model !== state.active) await selectModel(model);
    renderLegend(applyViewActions(viewer, actions));
  } catch (error) {
    showStatus(error.message);
  }
}

async function askQuestion(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  $("send-button").disabled = true;
  showStatus(null);
  chat.addUser(question);
  const pending = chat.addPending();
  const selection = { model: state.active, points: state.points, view: state.view };
  try {
    const result = await api.ask(question, selection, state.history);
    pending.remove();
    chat.addAnswer(result, () => show(result.view_actions));
    state.history = [
      ...state.history,
      { role: "user", content: question },
      { role: "assistant", content: result.answer.slice(0, MAX_TURN_CHARS) },
    ].slice(-MAX_HISTORY);
    if (result.view_actions.length) await show(result.view_actions);
  } catch (error) {
    pending.remove();
    chat.addError(error.message);
  } finally {
    state.busy = false;
    $("send-button").disabled = false;
  }
}

// ------------------------------------------------------------------ events

$("model-select").addEventListener("change", (event) => {
  selectModel(event.target.value).catch((error) => showStatus(error.message));
});

$("upload-button").addEventListener("click", () => $("upload-input").click());

$("upload-input").addEventListener("change", async (event) => {
  const [file] = event.target.files;
  event.target.value = ""; // allow uploading the same file again
  if (!file) return;
  showStatus(null);
  $("upload-button").disabled = true;
  try {
    const model = await api.uploadModel(file);
    await refreshModels();
    await selectModel(model.id);
  } catch (error) {
    showStatus(error.message);
  } finally {
    $("upload-button").disabled = false;
  }
});

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => viewer.setView(button.dataset.view));
});

$("clear-overlays").addEventListener("click", () => {
  viewer.clearOverlays();
  renderLegend([]);
});

$("clear-points").addEventListener("click", () => {
  state.points = [];
  renderPoints();
});

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("question");
  askQuestion(input.value);
  input.value = "";
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("ask-form").requestSubmit();
  }
});

$("suggestions").replaceChildren(
  ...SUGGESTIONS.map(([label, question]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.title = question;
    button.addEventListener("click", () => askQuestion(question));
    return button;
  }),
);

// -------------------------------------------------------------------- start

try {
  await refreshModels();
  if (state.models.length) await selectModel(state.models[0].id);
  else showStatus("No model is loaded on the server.");
} catch (error) {
  showStatus(error.message);
}
