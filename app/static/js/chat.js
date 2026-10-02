// Chat panel: questions, answers rendered as sanitised Markdown, and the tool-call trace.

import DOMPurify from "dompurify";
import { marked } from "marked";

// The answer is LLM output: it is rendered as Markdown, then sanitised. Images and
// forms are dropped altogether, so an answer cannot trigger a request to another site.
const SANITIZE = { FORBID_TAGS: ["img", "picture", "source", "form", "input", "style"] };

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** `slice_at(model="bridge", axis="z", value_mm=10)` */
function formatCall({ tool, args }) {
  const parts = Object.entries(args).map(([key, value]) => `${key}=${JSON.stringify(value)}`);
  return `${tool}(${parts.join(", ")})`;
}

export class Chat {
  /** @param {HTMLElement} container scrollable element receiving the messages */
  constructor(container) {
    this.container = container;
  }

  #add(node) {
    this.container.append(node);
    node.scrollIntoView({ block: "end" });
    return node;
  }

  addUser(text) {
    this.#add(element("div", "message user", text));
  }

  /** Placeholder shown while the server works; remove it with `.remove()`. */
  addPending() {
    return this.#add(element("div", "message pending", "Choosing tools…"));
  }

  addError(text) {
    this.#add(element("div", "message error", text));
  }

  /**
   * @param {{answer: string, trace: object[], view_actions: object[]}} result
   * @param {() => void} onShow called to draw this answer's view actions again
   */
  addAnswer({ answer, trace, view_actions: actions }, onShow) {
    const message = element("div", "message assistant");
    const body = element("div", "answer");
    body.innerHTML = DOMPurify.sanitize(marked.parse(answer || "_No answer._"), SANITIZE);
    message.append(body);

    const section = element("div", "trace");
    const title = element("div", "trace-title");
    const count = trace.length === 1 ? "1 tool call" : `${trace.length} tool calls`;
    title.append(element("span", "", trace.length ? count : "No tool call"));
    if (actions.length) {
      const show = element("button", "link", "Show in viewer");
      show.type = "button";
      show.addEventListener("click", onShow);
      title.append(show);
    }
    section.append(title);

    if (trace.length) {
      const list = element("ol");
      for (const step of trace) {
        const item = element("li", step.ok ? "" : "failed");
        const call = element("div", "call");
        call.append(element("code", "", formatCall(step)));
        item.append(call, element("div", "summary", step.summary));
        list.append(item);
      }
      section.append(list);
    }
    message.append(section);
    this.#add(message);
  }
}
