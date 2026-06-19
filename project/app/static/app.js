const state = {
  importSessionId: null,
  questions: [],
  testSessionId: null,
  currentTestQuestion: null,
  selectedOption: null,
  freeIndex: 0,
};

const $ = (id) => document.getElementById(id);

function show(id) {
  ["importView", "modeView", "testView", "freeView", "resultView"].forEach((view) => $(view).classList.add("hidden"));
  $(id).classList.remove("hidden");
}

function setLoading(elementId, isLoading) {
  $(elementId).classList.toggle("hidden", !isLoading);
}

function showError(message) {
  $("errorBox").textContent = message;
  $("errorBox").classList.remove("hidden");
}

function clearError() {
  $("errorBox").classList.add("hidden");
  $("errorBox").textContent = "";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

function renderRichText(value) {
  const parts = value.split(/```/g);
  return parts.map((part, index) => {
    if (index % 2 === 1) {
      return `<pre><code>${escapeHtml(part.trim())}</code></pre>`;
    }
    return `<p>${escapeHtml(part).replace(/\n/g, "<br>")}</p>`;
  }).join("");
}

function renderTags(container, question) {
  container.innerHTML = `
    <span class="tag">${escapeHtml(question.question_type)}</span>
    <span class="tag">${escapeHtml(question.topic)}</span>
    <span class="tag">${escapeHtml(question.difficulty)}</span>
  `;
}

function renderOptions(question) {
  $("testOptions").innerHTML = question.options.map((option, index) => `
    <div class="option" data-index="${index}">
      <strong>${String.fromCharCode(65 + index)}.</strong> ${escapeHtml(option)}
    </div>
  `).join("");
  document.querySelectorAll(".option").forEach((node) => {
    node.addEventListener("click", () => {
      state.selectedOption = Number(node.dataset.index);
      document.querySelectorAll(".option").forEach((item) => item.classList.remove("selected"));
      node.classList.add("selected");
    });
  });
}

function setProgress(prefix, current, total) {
  $(`${prefix}Progress`).textContent = `${current} / ${total}`;
  $(`${prefix}Bar`).style.width = `${Math.round((current / total) * 100)}%`;
}

function renderTestQuestion(question, answeredCount, total) {
  state.currentTestQuestion = question;
  state.selectedOption = null;
  renderTags($("testTags"), question);
  $("testQuestion").innerHTML = renderRichText(question.clean_text);
  renderOptions(question);
  setProgress("test", answeredCount + 1, total);
  show("testView");
}

function renderFreeQuestion() {
  const question = state.questions[state.freeIndex];
  renderTags($("freeTags"), question);
  $("freeQuestion").innerHTML = renderRichText(question.clean_text);
  $("freeAnswer").value = "";
  $("freeReview").classList.add("hidden");
  $("nextFreeBtn").classList.add("hidden");
  $("checkFreeBtn").disabled = false;
  setProgress("free", state.freeIndex + 1, state.questions.length);
  show("freeView");
}

function list(items) {
  if (!items || !items.length) return "<span class='muted'>Нет замечаний.</span>";
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function renderEvaluation(evaluation) {
  $("freeReview").innerHTML = `
    <div class="review-block"><h3>Оценка: ${evaluation.score}/100</h3><p>${escapeHtml(evaluation.verdict)}</p></div>
    <div class="review-block"><h3>Что верно</h3>${list(evaluation.correct_parts)}</div>
    <div class="review-block"><h3>Что неверно</h3>${list(evaluation.wrong_parts)}</div>
    <div class="review-block"><h3>Чего не хватает</h3>${list(evaluation.missing_parts)}</div>
    <div class="review-block"><h3>Как улучшить</h3>${list(evaluation.improvement_advice)}</div>
    <div class="review-block"><h3>Идеальный ответ</h3><p>${escapeHtml(evaluation.ideal_answer)}</p></div>
  `;
  $("freeReview").classList.remove("hidden");
}

async function renderResults(sessionId) {
  const data = await api(`/v1/results/${sessionId}`);
  const result = data.result;
  if (result.mode === "test") {
    $("resultContent").innerHTML = `
      <div class="result-block"><h3>${result.correct_count}/${result.total_questions} правильно (${result.percent}%)</h3></div>
      <div class="result-block"><h3>Ошибки</h3>${
        result.mistakes.length
          ? result.mistakes.map((item) => `<p><strong>${escapeHtml(item.question)}</strong><br>Ваш ответ: ${escapeHtml(item.selected_answer)}<br>Правильно: ${escapeHtml(item.correct_answer)}<br><span class="muted">${escapeHtml(item.explanation)}</span></p>`).join("")
          : "<p class='muted'>Ошибок нет.</p>"
      }</div>
    `;
  } else {
    $("resultContent").innerHTML = `
      <div class="result-block"><h3>Средний балл: ${result.average_score}/100</h3><p>Ответов: ${result.answered_count}</p></div>
      <div class="result-block"><h3>Сильные темы</h3>${list(result.strong_topics)}</div>
      <div class="result-block"><h3>Слабые темы</h3>${list(result.weak_topics)}</div>
      <div class="result-block"><h3>Рекомендации</h3>${list(result.recommendations)}</div>
    `;
  }
  show("resultView");
}

$("generateBtn").addEventListener("click", async () => {
  clearError();
  const raw = $("rawQuestions").value.trim();
  if (!raw) {
    showError("Вставьте вопросы перед генерацией.");
    return;
  }
  $("generateBtn").disabled = true;
  setLoading("loader", true);
  try {
    const data = await api("/v1/questions/import", {
      method: "POST",
      body: JSON.stringify({ raw_text: raw }),
    });
    state.importSessionId = data.session_id;
    state.questions = data.questions;
    $("questionCount").textContent = `Подготовлено вопросов: ${data.questions_count}`;
    $("testCount").max = data.questions_count;
    $("testCount").value = Math.min(Number($("testCount").value), data.questions_count);
    show("modeView");
  } catch (error) {
    showError(error.message);
  } finally {
    $("generateBtn").disabled = false;
    setLoading("loader", false);
  }
});

$("startTestBtn").addEventListener("click", async () => {
  const count = Number($("testCount").value || 1);
  const data = await api("/v1/test/start", {
    method: "POST",
    body: JSON.stringify({ source_session_id: state.importSessionId, question_count: count }),
  });
  state.testSessionId = data.session_id;
  renderTestQuestion(data.current_question, 0, data.total_questions);
});

$("answerTestBtn").addEventListener("click", async () => {
  if (state.selectedOption === null) return;
  const data = await api("/v1/test/answer", {
    method: "POST",
    body: JSON.stringify({
      session_id: state.testSessionId,
      question_id: state.currentTestQuestion.id,
      selected_index: state.selectedOption,
    }),
  });
  if (data.completed) {
    await renderResults(state.testSessionId);
  } else {
    renderTestQuestion(data.next_question, data.answered_count, data.total_questions);
  }
});

$("startFreeBtn").addEventListener("click", () => {
  state.freeIndex = 0;
  renderFreeQuestion();
});

$("checkFreeBtn").addEventListener("click", async () => {
  const question = state.questions[state.freeIndex];
  const answer = $("freeAnswer").value.trim();
  if (!answer) return;
  $("checkFreeBtn").disabled = true;
  setLoading("freeLoader", true);
  try {
    const data = await api("/v1/free/answer", {
      method: "POST",
      body: JSON.stringify({ session_id: state.importSessionId, question_id: question.id, answer }),
    });
    renderEvaluation(data.evaluation);
    $("nextFreeBtn").classList.remove("hidden");
    $("nextFreeBtn").textContent = state.freeIndex + 1 >= state.questions.length ? "Итоги" : "Следующий вопрос";
  } catch (error) {
    renderEvaluation({ score: 0, verdict: error.message, correct_parts: [], wrong_parts: [], missing_parts: [], improvement_advice: [], ideal_answer: "" });
  } finally {
    setLoading("freeLoader", false);
  }
});

$("nextFreeBtn").addEventListener("click", async () => {
  state.freeIndex += 1;
  if (state.freeIndex >= state.questions.length) {
    await renderResults(state.importSessionId);
  } else {
    renderFreeQuestion();
  }
});

$("restartBtn").addEventListener("click", () => {
  state.importSessionId = null;
  state.questions = [];
  state.testSessionId = null;
  show("importView");
});

fetch("/health/live").then(() => {
  $("apiStatus").textContent = "API online";
}).catch(() => {
  $("apiStatus").textContent = "API offline";
});
