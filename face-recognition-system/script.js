const fileInput = document.querySelector("#fileInput");
const uploadButton = document.querySelector("#uploadButton");
const dropZone = document.querySelector("#dropZone");
const imagePreview = document.querySelector("#imagePreview");
const videoPreview = document.querySelector("#videoPreview");
const overlay = document.querySelector("#overlay");
const emptyState = document.querySelector("#emptyState");
const detectButton = document.querySelector("#detectButton");
const pauseButton = document.querySelector("#pauseButton");
const resetButton = document.querySelector("#resetButton");
const engineStatus = document.querySelector("#engineStatus");
const faceCount = document.querySelector("#faceCount");
const libraryCount = document.querySelector("#libraryCount");
const recordCount = document.querySelector("#recordCount");
const detectState = document.querySelector("#detectState");
const faceList = document.querySelector("#faceList");
const lastRun = document.querySelector("#lastRun");
const fileTypeMetric = document.querySelector("#fileTypeMetric");
const supportHint = document.querySelector("#supportHint");
const personForm = document.querySelector("#personForm");
const personName = document.querySelector("#personName");
const personCode = document.querySelector("#personCode");
const personDept = document.querySelector("#personDept");
const personNote = document.querySelector("#personNote");
const registerFiles = document.querySelector("#registerFiles");
const registerButton = document.querySelector("#registerButton");
const clearFormButton = document.querySelector("#clearFormButton");
const registrationStatus = document.querySelector("#registrationStatus");
const registrationHint = document.querySelector("#registrationHint");
const libraryList = document.querySelector("#libraryList");
const librarySummary = document.querySelector("#librarySummary");
const recordList = document.querySelector("#recordList");
const recordSummary = document.querySelector("#recordSummary");

const STORAGE_KEYS = {
  people: "frs.people.v1",
  records: "frs.records.v1",
};

const MAX_RECORDS = 60;
const MATCH_THRESHOLD = 0.84;
const FEATURE_SIZE = 24;
const THUMB_SIZE = 160;

let detector = null;
let currentMode = null;
let currentObjectUrl = null;
let videoTimer = null;
let lastRecognition = [];
let state = {
  people: [],
  records: [],
};

function uid() {
  if (window.crypto?.randomUUID) return crypto.randomUUID();
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEYS.people, JSON.stringify(state.people));
  localStorage.setItem(STORAGE_KEYS.records, JSON.stringify(state.records.slice(0, MAX_RECORDS)));
}

function loadState() {
  state.people = loadJson(STORAGE_KEYS.people, []).map((person) => ({
    ...person,
    faces: Array.isArray(person.faces) ? person.faces : [],
  }));
  state.records = loadJson(STORAGE_KEYS.records, []);
}

function nowLabel(date = new Date()) {
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

async function setupDetector() {
  if (!("FaceDetector" in window)) {
    engineStatus.textContent = "当前浏览器不支持";
    engineStatus.classList.add("unsupported");
    detectState.textContent = "不可用";
    detectButton.disabled = true;
    supportHint.textContent = "当前浏览器不支持 FaceDetector。请使用新版 Chrome 或 Edge。";
    return;
  }

  detector = new FaceDetector({
    fastMode: false,
    maxDetectedFaces: 20,
  });
  engineStatus.textContent = "识别引擎已就绪";
  detectState.textContent = "待识别";
  supportHint.hidden = false;
}

function clearCanvas() {
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, overlay.width, overlay.height);
}

function getSourceSize(source) {
  return {
    width: source.videoWidth || source.naturalWidth || source.clientWidth || 0,
    height: source.videoHeight || source.naturalHeight || source.clientHeight || 0,
  };
}

function setCanvasSize(source) {
  const { width, height } = getSourceSize(source);
  if (!width || !height) return;
  overlay.width = width;
  overlay.height = height;
  overlay.style.width = `${source.clientWidth}px`;
  overlay.style.height = `${source.clientHeight}px`;
}

function clampRect(rect, sourceWidth, sourceHeight, paddingRatio = 0.18) {
  const paddingX = rect.width * paddingRatio;
  const paddingY = rect.height * paddingRatio;
  const x = Math.max(0, Math.floor(rect.x - paddingX));
  const y = Math.max(0, Math.floor(rect.y - paddingY));
  const width = Math.min(sourceWidth - x, Math.ceil(rect.width + paddingX * 2));
  const height = Math.min(sourceHeight - y, Math.ceil(rect.height + paddingY * 2));
  return { x, y, width: Math.max(1, width), height: Math.max(1, height) };
}

function drawFaces(source, faces, labels = []) {
  setCanvasSize(source);
  const context = overlay.getContext("2d");
  const sourceSize = getSourceSize(source);
  const scaleX = overlay.width / sourceSize.width;
  const scaleY = overlay.height / sourceSize.height;

  context.clearRect(0, 0, overlay.width, overlay.height);
  context.lineWidth = Math.max(2, overlay.width / 220);
  context.strokeStyle = "#35e09b";
  context.fillStyle = "rgba(53, 224, 155, 0.18)";
  context.font = `${Math.max(13, overlay.width / 46)}px Arial`;

  faces.forEach((face, index) => {
    const { x, y, width, height } = face.boundingBox;
    const boxX = x * scaleX;
    const boxY = y * scaleY;
    const boxWidth = width * scaleX;
    const boxHeight = height * scaleY;
    const label = labels[index] || `人脸 ${index + 1}`;

    context.fillRect(boxX, boxY, boxWidth, boxHeight);
    context.strokeRect(boxX, boxY, boxWidth, boxHeight);
    const textWidth = context.measureText(label).width + 16;
    const tagX = boxX;
    const tagY = Math.max(0, boxY - 26);
    context.fillStyle = "#10211b";
    context.fillRect(tagX, tagY, Math.min(textWidth, overlay.width - tagX), 22);
    context.fillStyle = "#ffffff";
    context.fillText(label, tagX + 8, tagY + 16);
    context.fillStyle = "rgba(53, 224, 155, 0.18)";
  });
}

function normalizeVector(vector) {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / magnitude);
}

function extractFeatureVector(source, face) {
  const sourceSize = getSourceSize(source);
  const rect = clampRect(face.boundingBox, sourceSize.width, sourceSize.height);
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = FEATURE_SIZE;
  cropCanvas.height = FEATURE_SIZE;
  const ctx = cropCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, rect.x, rect.y, rect.width, rect.height, 0, 0, FEATURE_SIZE, FEATURE_SIZE);

  const { data } = ctx.getImageData(0, 0, FEATURE_SIZE, FEATURE_SIZE);
  const vector = [];
  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let graySum = 0;
  let graySqSum = 0;

  for (let index = 0; index < data.length; index += 4) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

    vector.push(gray);
    rSum += r;
    gSum += g;
    bSum += b;
    graySum += gray;
    graySqSum += gray * gray;
  }

  const pixelCount = data.length / 4;
  const avgR = rSum / pixelCount / 255;
  const avgG = gSum / pixelCount / 255;
  const avgB = bSum / pixelCount / 255;
  const avgGray = graySum / pixelCount;
  const variance = Math.max(0, graySqSum / pixelCount - avgGray * avgGray);

  vector.push(avgR, avgG, avgB, avgGray, Math.sqrt(variance));

  return {
    vector: normalizeVector(vector),
    thumbnail: cropCanvas.toDataURL("image/jpeg", 0.82),
  };
}

function similarity(a, b) {
  const length = Math.min(a.length, b.length);
  let score = 0;
  for (let index = 0; index < length; index += 1) {
    score += a[index] * b[index];
  }
  return Math.max(0, Math.min(1, score));
}

function sourceLabel(file) {
  return file ? file.name : "未知来源";
}

async function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("图片加载失败"));
    };
    image.src = url;
  });
}

async function detectFaces(source) {
  if (!detector) return [];
  const faces = await detector.detect(source);
  return faces.sort((left, right) => right.boundingBox.width * right.boundingBox.height - left.boundingBox.width * left.boundingBox.height);
}

function findBestMatch(featureVector) {
  let best = null;

  state.people.forEach((person) => {
    person.faces.forEach((face) => {
      const score = similarity(featureVector, face.vector);
      if (!best || score > best.score) {
        best = {
          person,
          face,
          score,
        };
      }
    });
  });

  return best;
}

function formatMatchLabel(match) {
  if (!match || match.score < MATCH_THRESHOLD) return "未知人员";
  const name = match.person.name || "未知人员";
  return `${name} ${(match.score * 100).toFixed(1)}%`;
}

function renderRecognitionResults(faces, matches, sourceName) {
  faceCount.textContent = String(faces.length);
  lastRun.textContent = nowLabel();

  if (faces.length === 0) {
    faceList.innerHTML = '<div class="empty-row"><strong>未检测到人脸</strong><span>请尝试更清晰、正脸更多的素材。</span></div>';
    return;
  }

  faceList.innerHTML = faces
    .map((face, index) => {
      const match = matches[index];
      const recognized = match && match.score >= MATCH_THRESHOLD;
      const name = recognized ? match.person.name : "未知人员";
      const code = recognized ? match.person.personCode || "-" : "-";
      const scoreText = recognized ? `${(match.score * 100).toFixed(2)}%` : "低于阈值";
      return `
        <div class="face-row ${recognized ? "matched" : "unknown"}">
          <strong>${name}</strong>
          <span>相似度：${scoreText}</span>
          <span>编号：${code}</span>
          <span>来源：${sourceName}</span>
        </div>
      `;
    })
    .join("");
}

function saveRecognitionRecord(match, sourceName, faceIndex) {
  const recognized = match && match.score >= MATCH_THRESHOLD;
  state.records.unshift({
    id: uid(),
    createdAt: new Date().toISOString(),
    sourceName,
    faceIndex,
    personId: recognized ? match.person.id : null,
    personName: recognized ? match.person.name : "未知人员",
    personCode: recognized ? match.person.personCode || "" : "",
    similarity: recognized ? Number((match.score * 100).toFixed(2)) : null,
    status: recognized ? "recognized" : "unknown",
  });
  state.records = state.records.slice(0, MAX_RECORDS);
}

function renderLibrary() {
  libraryCount.textContent = String(state.people.length);
  const photoTotal = state.people.reduce((sum, person) => sum + person.faces.length, 0);
  librarySummary.textContent = state.people.length
    ? `${state.people.length} 人员 / ${photoTotal} 张注册照片`
    : "暂无人员";

  if (!state.people.length) {
    libraryList.innerHTML = '<div class="empty-row"><strong>人脸库为空</strong><span>先在上方录入人员并上传注册照片。</span></div>';
    return;
  }

  libraryList.innerHTML = state.people
    .slice()
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
    .map((person) => {
      const photoCount = person.faces.length;
      const mainFace = person.faces[0];
      const preview = mainFace?.thumbnail || "";
      return `
        <article class="library-item">
          <div class="library-preview">
            ${preview ? `<img src="${preview}" alt="${person.name} 的人脸缩略图" />` : ""}
          </div>
          <div class="library-meta">
            <strong>${person.name}</strong>
            <span>编号：${person.personCode || "-"}</span>
            <span>部门：${person.department || "-"}</span>
            <span>照片：${photoCount} 张</span>
            <span>备注：${person.note || "-"}</span>
          </div>
          <button class="danger" type="button" data-delete-person="${person.id}">删除人员数据</button>
        </article>
      `;
    })
    .join("");

  libraryList.querySelectorAll("[data-delete-person]").forEach((button) => {
    button.addEventListener("click", () => deletePerson(button.dataset.deletePerson));
  });
}

function renderRecords() {
  recordCount.textContent = String(state.records.length);
  recordSummary.textContent = state.records.length ? `最近 ${Math.min(state.records.length, MAX_RECORDS)} 条` : "暂无记录";

  if (!state.records.length) {
    recordList.innerHTML = '<div class="empty-row"><strong>暂无识别记录</strong><span>识别成功或未知人员都会被保存到这里。</span></div>';
    return;
  }

  recordList.innerHTML = state.records
    .slice(0, 12)
    .map((record) => {
      const label = record.status === "recognized" ? `${record.personName} ${record.similarity.toFixed(2)}%` : "未知人员";
      const code = record.personCode || "-";
      return `
        <div class="record-row ${record.status}">
          <strong>${label}</strong>
          <span>编号：${code}</span>
          <span>时间：${nowLabel(new Date(record.createdAt))}</span>
          <span>来源：${record.sourceName}</span>
        </div>
      `;
    })
    .join("");
}

function renderAll() {
  renderLibrary();
  renderRecords();
}

function getOrCreatePerson(formData) {
  const { name, code } = formData;
  const targetCode = code.trim();
  const targetName = name.trim();
  let person = null;

  if (targetCode) {
    person = state.people.find((item) => item.personCode === targetCode) || null;
  }

  if (!person) {
    person = state.people.find((item) => item.name === targetName && !targetCode) || null;
  }

  if (!person) {
    person = {
      id: uid(),
      name: targetName,
      personCode: targetCode,
      department: formData.department.trim(),
      note: formData.note.trim(),
      createdAt: new Date().toISOString(),
      faces: [],
    };
    state.people.unshift(person);
  } else {
    person.name = targetName;
    person.personCode = targetCode || person.personCode;
    person.department = formData.department.trim();
    person.note = formData.note.trim();
  }

  return person;
}

async function registerFaces(event) {
  event.preventDefault();

  if (!detector) {
    registrationStatus.textContent = "当前浏览器不支持";
    return;
  }

  const name = personName.value.trim();
  if (!name) {
    registrationStatus.textContent = "姓名不能为空";
    return;
  }

  const files = Array.from(registerFiles.files || []);
  if (!files.length) {
    registrationStatus.textContent = "请选择注册照片";
    return;
  }

  registerButton.disabled = true;
  registrationStatus.textContent = "正在提取特征";

  let imported = 0;
  let skipped = 0;
  const person = getOrCreatePerson({
    name,
    code: personCode.value,
    department: personDept.value,
    note: personNote.value,
  });

  for (const file of files) {
    try {
      const image = await loadImageFromFile(file);
      const faces = await detectFaces(image);
      if (!faces.length) {
        skipped += 1;
        continue;
      }

      const face = faces[0];
      const feature = extractFeatureVector(image, face);
      person.faces.push({
        id: uid(),
        fileName: file.name,
        createdAt: new Date().toISOString(),
        thumbnail: feature.thumbnail,
        vector: feature.vector,
      });
      imported += 1;
    } catch {
      skipped += 1;
    }
  }

  person.faces = person.faces.slice(-20);
  saveState();
  renderAll();
  registrationStatus.textContent = `已注册 ${imported} 张，跳过 ${skipped} 张`;
  registrationHint.textContent = "特征已写入本地人脸库。";
  personForm.reset();
  registerButton.disabled = false;
}

function deletePerson(personId) {
  const person = state.people.find((item) => item.id === personId);
  if (!person) return;
  const confirmed = window.confirm(`确定删除 ${person.name} 的所有人脸数据吗？`);
  if (!confirmed) return;

  state.people = state.people.filter((item) => item.id !== personId);
  state.records = state.records.filter((record) => record.personId !== personId);
  saveState();
  renderAll();
}

function stopVideoScan() {
  if (videoTimer) {
    clearInterval(videoTimer);
    videoTimer = null;
  }
  pauseButton.disabled = true;
}

function clearRecognitionState() {
  stopVideoScan();
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }

  currentMode = null;
  fileInput.value = "";
  imagePreview.hidden = true;
  videoPreview.hidden = true;
  imagePreview.removeAttribute("src");
  videoPreview.removeAttribute("src");
  emptyState.hidden = false;
  detectButton.disabled = true;
  pauseButton.disabled = true;
  faceCount.textContent = "0";
  fileTypeMetric.textContent = "未上传";
  detectState.textContent = detector ? "待识别" : "不可用";
  faceList.innerHTML = "";
  lastRun.textContent = "暂无记录";
  clearCanvas();
}

async function runRecognition(source, sourceName) {
  if (!detector || !source) return;

  try {
    detectState.textContent = "识别中";
    const faces = await detectFaces(source);
    const matches = [];

    for (const face of faces) {
      const feature = extractFeatureVector(source, face);
      matches.push(findBestMatch(feature.vector));
    }

    lastRecognition = matches.map((match, index) => {
      const recognized = match && match.score >= MATCH_THRESHOLD;
      return {
        label: recognized ? `${match.person.name} ${(match.score * 100).toFixed(1)}%` : "未知人员",
        score: match ? Number((match.score * 100).toFixed(2)) : 0,
        recognized,
      };
    });

    drawFaces(source, faces, lastRecognition.map((item) => item.label));
    renderRecognitionResults(faces, matches, sourceName);

    faces.forEach((_, index) => {
      saveRecognitionRecord(matches[index], sourceName, index + 1);
    });

    saveState();
    renderRecords();
    detectState.textContent = faces.length ? "已完成" : "未检测到人脸";
  } catch (error) {
    detectState.textContent = "识别失败";
    faceList.innerHTML = `<div class="empty-row"><strong>识别失败</strong><span>${error.message}</span></div>`;
  }
}

function loadFile(file) {
  if (!file) return;

  clearRecognitionState();
  currentObjectUrl = URL.createObjectURL(file);
  emptyState.hidden = true;
  detectButton.disabled = !detector;
  supportHint.hidden = Boolean(detector);

  if (file.type.startsWith("image/")) {
    currentMode = "image";
    fileTypeMetric.textContent = "照片";
    imagePreview.src = currentObjectUrl;
    imagePreview.hidden = false;
    imagePreview.onload = () => {
      setCanvasSize(imagePreview);
      runRecognition(imagePreview, sourceLabel(file));
    };
    return;
  }

  if (file.type.startsWith("video/")) {
    currentMode = "video";
    fileTypeMetric.textContent = "视频";
    videoPreview.src = currentObjectUrl;
    videoPreview.hidden = false;
    videoPreview.onloadedmetadata = () => {
      setCanvasSize(videoPreview);
      videoPreview.play().catch(() => {});
      startVideoScan(file);
    };
    return;
  }

  detectState.textContent = "格式不支持";
}

function startVideoScan(file) {
  if (currentMode !== "video") return;
  stopVideoScan();
  runRecognition(videoPreview, sourceLabel(file));
  videoTimer = setInterval(() => {
    if (!videoPreview.paused && !videoPreview.ended) {
      runRecognition(videoPreview, sourceLabel(file));
    }
  }, 1200);
  pauseButton.disabled = false;
}

function clearRegistrationForm() {
  personForm.reset();
  registrationStatus.textContent = "等待录入";
  registrationHint.textContent = "每张照片会提取本地特征向量并写入人脸库。建议使用正脸、光线稳定的照片。";
}

async function manualRecognition() {
  if (!currentMode) return;
  const source = currentMode === "video" ? videoPreview : imagePreview;
  const currentFile = fileInput.files?.[0];
  await runRecognition(source, sourceLabel(currentFile));
}

uploadButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));

detectButton.addEventListener("click", manualRecognition);

pauseButton.addEventListener("click", () => {
  stopVideoScan();
  detectState.textContent = "已暂停";
});

resetButton.addEventListener("click", clearRecognitionState);

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  loadFile(event.dataTransfer.files[0]);
});

personForm.addEventListener("submit", registerFaces);
clearFormButton.addEventListener("click", clearRegistrationForm);

window.addEventListener("resize", () => {
  const source = currentMode === "video" ? videoPreview : imagePreview;
  if (currentMode) setCanvasSize(source);
});

loadState();
renderAll();
setupDetector();
