import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js";
import {
  addDoc,
  collection,
  deleteDoc,
  doc,
  getFirestore,
  limit,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.13.2/firebase-firestore.js";
import {
  deleteObject,
  getDownloadURL,
  getStorage,
  ref as storageRef,
  uploadBytes
} from "https://www.gstatic.com/firebasejs/10.13.2/firebase-storage.js";

const firebaseConfig = {
  apiKey: "AIzaSyDR_Bt2tltKxSfLSvmlW5mQ0uwxlmafy3w",
  authDomain: "airacare-animal-safety.firebaseapp.com",
  projectId: "airacare-animal-safety",
  storageBucket: "airacare-animal-safety.firebasestorage.app",
  messagingSenderId: "934286949654",
  appId: "1:934286949654:android:a0c98c97bd4568e95ee437"
};

const DETECTION_INTERVAL_MS = 900;
const MAX_CAPTURE_WIDTH = 512;
const DETECTION_TIMEOUT_MS = 30000;
const PET_LABELS = new Set(["dog", "cat"]);
const STABLE_PET_FRAMES = 2;
const CAPTURE_COOLDOWN_MS = 2500;
const DEFAULT_API_BASE_URL = "https://airacare-web-backend.onrender.com";
const API_BASE_URL = normalizeApiBaseUrl(
  new URLSearchParams(window.location.search).get("api") || DEFAULT_API_BASE_URL
);

const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore(firebaseApp);
const storage = getStorage(firebaseApp);

const video = document.getElementById("cameraVideo");
const overlay = document.getElementById("overlayCanvas");
const overlayCtx = overlay.getContext("2d");
const captureCanvas = document.getElementById("captureCanvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });
const petCropCanvas = document.getElementById("petCropCanvas");
const petCropCtx = petCropCanvas.getContext("2d", { willReadFrequently: true });
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const captureStoreButton = document.getElementById("captureStoreButton");
const captureStoreStatus = document.getElementById("captureStoreStatus");
const statusPill = document.getElementById("statusPill");
const warningBanner = document.getElementById("warningBanner");
const latestLabel = document.getElementById("latestLabel");
const latestDistance = document.getElementById("latestDistance");
const latestRisk = document.getElementById("latestRisk");
const firebaseStatus = document.getElementById("firebaseStatus");
const modelStatus = document.getElementById("modelStatus");
const historyList = document.getElementById("historyList");
const collectionList = document.getElementById("collectionList");
const collectionTabs = Array.from(document.querySelectorAll(".collection-tab"));
const captureModal = document.getElementById("captureModal");
const captureModalTitle = document.getElementById("captureModalTitle");
const capturePreviewImage = document.getElementById("capturePreviewImage");
const capturePreviewLabel = document.getElementById("capturePreviewLabel");
const capturePreviewConfidence = document.getElementById("capturePreviewConfidence");
const capturePreviewTime = document.getElementById("capturePreviewTime");
const capturePreviewNote = document.getElementById("capturePreviewNote");
const cancelCaptureButton = document.getElementById("cancelCaptureButton");
const saveCaptureButton = document.getElementById("saveCaptureButton");

let stream = null;
let running = false;
let loopTimer = null;
let latestPosition = null;
let lastSavedBySignature = new Map();
let consecutiveDetectionErrors = 0;
let detectionInFlight = false;
let latestDetectionResult = null;
let stablePetCandidate = null;
let petStability = { key: null, count: 0 };
let captureInFlight = false;
let saveInFlight = false;
let lastCaptureAt = 0;
let pendingCapture = null;
let collectionFilter = "all";
let collectionItems = [];
let anonymousId = getAnonymousId();

startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
captureStoreButton.addEventListener("click", captureAndStorePet);
cancelCaptureButton.addEventListener("click", closeCaptureModal);
saveCaptureButton.addEventListener("click", savePendingCapture);
collectionTabs.forEach((tab) => {
  tab.addEventListener("click", () => setCollectionFilter(tab.dataset.filter || "all"));
});
window.addEventListener("resize", resizeOverlay);

listenToRecentDetections();
listenToCapturedPets();
checkBackend();
console.info("[Airacare] Backend base URL:", API_BASE_URL);

async function checkBackend() {
  const endpoint = `${API_BASE_URL}/api/health`;
  console.info("[Airacare] Health endpoint:", endpoint);
  try {
    const response = await fetch(endpoint, { cache: "no-store" });
    const text = await response.text();
    console.info("[Airacare] Health status:", response.status);
    console.info("[Airacare] Health response:", text);
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    const data = JSON.parse(text);
    modelStatus.textContent = data.modelReady === false
      ? "Backend online, model error"
      : data.usingPretrainedFallback
      ? "Pretrained fallback loaded"
      : "Airacare model loaded";
  } catch (error) {
    modelStatus.textContent = "Backend offline";
    console.error(error);
  }
}

async function start() {
  try {
    setStatus("Opening camera");
    startButton.disabled = true;
    await startLocationWatch();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    running = true;
    stopButton.disabled = false;
    setStatus("Detecting");
    detectLoop();
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Camera failed");
    startButton.disabled = false;
    stopButton.disabled = true;
  }
}

function stop() {
  running = false;
  if (loopTimer) window.clearTimeout(loopTimer);
  loopTimer = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  warningBanner.hidden = true;
  latestDetectionResult = null;
  stablePetCandidate = null;
  petStability = { key: null, count: 0 };
  updateCaptureStoreButton();
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Stopped");
}

async function detectLoop() {
  if (!running) return;
  if (detectionInFlight) {
    loopTimer = window.setTimeout(detectLoop, DETECTION_INTERVAL_MS);
    return;
  }
  detectionInFlight = true;
  try {
    const image = captureFrame();
    const endpoint = `${API_BASE_URL}/api/detect`;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), DETECTION_TIMEOUT_MS);
    console.info("[Airacare] Detection endpoint:", endpoint);
    const response = await fetch(
      endpoint,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          image,
          ...freshPosition(),
          deviceType: detectDeviceType(),
          captureWidth: captureCanvas.width,
          captureHeight: captureCanvas.height,
          videoWidth: video.videoWidth || null,
          videoHeight: video.videoHeight || null,
          screenOrientation: screen.orientation?.type || null,
          devicePixelRatio: window.devicePixelRatio || 1
        })
      }
    );
    window.clearTimeout(timeout);
    const text = await response.text();
    console.info("[Airacare] Detection status:", response.status);
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    const result = JSON.parse(text);
    console.info(
      "[Airacare] Detection result:",
      `${result.detections?.length || 0} objects`,
      `${result.processingTimeMs || "?"}ms`
    );
    consecutiveDetectionErrors = 0;
    setStatus("Detecting");
    latestDetectionResult = result;
    drawDetections(result);
    updateLatest(result);
    updateStablePetCandidate(result);
    saveDetections(result).catch((saveError) => {
      firebaseStatus.textContent = "Error";
      console.error(saveError);
    });
  } catch (error) {
    consecutiveDetectionErrors += 1;
    console.error(error);
    setStatus(consecutiveDetectionErrors >= 3 ? "Backend busy" : "Retrying");
  } finally {
    detectionInFlight = false;
    loopTimer = window.setTimeout(detectLoop, DETECTION_INTERVAL_MS);
  }
}

function captureFrame() {
  const sourceWidth = video.videoWidth || 640;
  const sourceHeight = video.videoHeight || 480;
  const scale = Math.min(1, MAX_CAPTURE_WIDTH / sourceWidth);
  captureCanvas.width = Math.round(sourceWidth * scale);
  captureCanvas.height = Math.round(sourceHeight * scale);
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.62);
}

function drawDetections(result) {
  resizeOverlay();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const scaleX = overlay.width / Math.max(1, result.frameWidth);
  const scaleY = overlay.height / Math.max(1, result.frameHeight);

  for (const detection of result.detections || []) {
    const box = detection.boundingBox;
    const x = box.left * scaleX;
    const y = box.top * scaleY;
    const width = (box.right - box.left) * scaleX;
    const height = (box.bottom - box.top) * scaleY;
    const isWarning = detection.warningTriggered;
    overlayCtx.strokeStyle = isWarning ? "#ff4d5d" : "#35c88a";
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeRect(x, y, width, height);

    const label = `${detection.label.toUpperCase()} ${(detection.confidence * 100).toFixed(0)}% ${formatDistance(detection.distanceMeters)}`;
    overlayCtx.font = "16px Arial";
    const textWidth = overlayCtx.measureText(label).width + 12;
    const textY = Math.max(24, y);
    overlayCtx.fillStyle = isWarning ? "#ff4d5d" : "#35c88a";
    overlayCtx.fillRect(x, textY - 22, textWidth, 22);
    overlayCtx.fillStyle = "#061014";
    overlayCtx.fillText(label, x + 6, textY - 6);
  }
}

function updateLatest(result) {
  const detections = result.detections || [];
  if (detections.length === 0) {
    latestLabel.textContent = "None";
    latestDistance.textContent = "N/A";
    latestRisk.textContent = result.sceneRisk || "N/A";
    warningBanner.hidden = true;
    return;
  }

  const primary = detections[0];
  latestLabel.textContent = primary.label;
  latestDistance.textContent = formatDistance(primary.distanceMeters);
  latestRisk.textContent = primary.riskLevel;

  if (result.warning?.message) {
    warningBanner.textContent = result.warning.message;
    warningBanner.hidden = false;
    playWarningTone();
  } else {
    warningBanner.hidden = true;
  }
}

function updateStablePetCandidate(result) {
  const pet = selectPrimaryPet(result?.detections || []);
  if (!pet) {
    stablePetCandidate = null;
    petStability = { key: null, count: 0 };
    updateCaptureStoreButton();
    return;
  }

  const key = `${pet.trackId ?? "no-track"}:${pet.label}`;
  if (petStability.key === key) {
    petStability.count += 1;
  } else {
    petStability = { key, count: 1 };
  }

  stablePetCandidate = petStability.count >= STABLE_PET_FRAMES ? pet : null;
  updateCaptureStoreButton();
}

function selectPrimaryPet(detections) {
  return detections
    .filter((detection) => PET_LABELS.has(String(detection.label || "").toLowerCase()))
    .sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0] || null;
}

function updateCaptureStoreButton() {
  const canCapture = Boolean(stablePetCandidate && running && !captureInFlight);
  captureStoreButton.hidden = !stablePetCandidate;
  captureStoreButton.disabled = !canCapture;
  if (stablePetCandidate) {
    const label = String(stablePetCandidate.label || "pet").toLowerCase();
    captureStoreButton.textContent = captureInFlight ? "Creating sticker..." : `Capture & Store ${label}`;
  } else {
    captureStoreButton.textContent = "Capture & Store";
  }
}

async function captureAndStorePet() {
  if (!stablePetCandidate || !latestDetectionResult || captureInFlight) return;
  if (Date.now() - lastCaptureAt < CAPTURE_COOLDOWN_MS) {
    captureStoreStatus.textContent = "Please wait before capturing again.";
    return;
  }

  captureInFlight = true;
  lastCaptureAt = Date.now();
  updateCaptureStoreButton();
  captureStoreStatus.textContent = "Creating sticker...";

  try {
    const cropDataUrl = cropPetFromCurrentFrame(stablePetCandidate, latestDetectionResult);
    const endpoint = `${API_BASE_URL}/api/remove-background`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: cropDataUrl,
        label: stablePetCandidate.label
      })
    });
    const text = await response.text();
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    const backgroundResult = JSON.parse(text);
    if (!backgroundResult.ok || !backgroundResult.image) {
      throw new Error(backgroundResult.error || "Background removal failed");
    }

    pendingCapture = {
      label: String(stablePetCandidate.label || "").toLowerCase(),
      confidence: Number(stablePetCandidate.confidence || 0),
      imageDataUrl: backgroundResult.image,
      backgroundRemoved: Boolean(backgroundResult.backgroundRemoved),
      backgroundRemovalMethod: backgroundResult.method || "unknown",
      boundingBox: { ...stablePetCandidate.boundingBox },
      distanceMeters: stablePetCandidate.distanceMeters ?? null,
      riskLevel: stablePetCandidate.riskLevel || null,
      cameraSource: stablePetCandidate.cameraSource || "web_camera_backend",
      trackId: stablePetCandidate.trackId ?? null,
      capturedAtEpochMillis: Date.now()
    };
    showCapturePreview(pendingCapture);
    captureStoreStatus.textContent = backgroundResult.backgroundRemoved
      ? "Sticker ready. Review before saving."
      : "Crop ready. Background removal fallback was used.";
  } catch (error) {
    console.error(error);
    captureStoreStatus.textContent = "Could not create sticker. Try again.";
  } finally {
    captureInFlight = false;
    updateCaptureStoreButton();
  }
}

function cropPetFromCurrentFrame(detection, result) {
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error("Video frame is unavailable");
  }
  const box = detection.boundingBox;
  if (!box) throw new Error("Bounding box is unavailable");

  const frameWidth = Math.max(1, Number(result.frameWidth || video.videoWidth));
  const frameHeight = Math.max(1, Number(result.frameHeight || video.videoHeight));
  const scaleX = video.videoWidth / frameWidth;
  const scaleY = video.videoHeight / frameHeight;

  let left = Number(box.left) * scaleX;
  let top = Number(box.top) * scaleY;
  let right = Number(box.right) * scaleX;
  let bottom = Number(box.bottom) * scaleY;
  if (![left, top, right, bottom].every(Number.isFinite) || right <= left || bottom <= top) {
    throw new Error("Invalid crop coordinates");
  }

  const padding = Math.max(right - left, bottom - top) * 0.08;
  left = clamp(left - padding, 0, video.videoWidth - 1);
  top = clamp(top - padding, 0, video.videoHeight - 1);
  right = clamp(right + padding, left + 1, video.videoWidth);
  bottom = clamp(bottom + padding, top + 1, video.videoHeight);

  const cropWidth = Math.max(1, Math.round(right - left));
  const cropHeight = Math.max(1, Math.round(bottom - top));
  petCropCanvas.width = cropWidth;
  petCropCanvas.height = cropHeight;
  petCropCtx.clearRect(0, 0, cropWidth, cropHeight);
  petCropCtx.drawImage(
    video,
    Math.round(left),
    Math.round(top),
    cropWidth,
    cropHeight,
    0,
    0,
    cropWidth,
    cropHeight
  );
  return petCropCanvas.toDataURL("image/jpeg", 0.88);
}

function showCapturePreview(capture) {
  captureModalTitle.textContent = `Captured ${capitalize(capture.label)}`;
  capturePreviewImage.src = capture.imageDataUrl;
  capturePreviewLabel.textContent = capture.label;
  capturePreviewConfidence.textContent = `${(capture.confidence * 100).toFixed(0)}%`;
  capturePreviewTime.textContent = formatTime(capture.capturedAtEpochMillis);
  capturePreviewNote.textContent = capture.backgroundRemoved
    ? "Transparent sticker created."
    : "Fallback crop created because background removal was uncertain.";
  saveCaptureButton.disabled = false;
  captureModal.hidden = false;
}

function closeCaptureModal() {
  if (saveInFlight) return;
  captureModal.hidden = true;
  pendingCapture = null;
}

async function savePendingCapture() {
  if (!pendingCapture || saveInFlight) return;
  saveInFlight = true;
  saveCaptureButton.disabled = true;
  capturePreviewNote.textContent = "Saving to collection...";
  try {
    const blob = await dataUrlToBlob(pendingCapture.imageDataUrl);
    const storagePath = `captured_pets/${anonymousId}/${pendingCapture.capturedAtEpochMillis}_${pendingCapture.label}.png`;
    const imageReference = storageRef(storage, storagePath);
    await uploadBytes(imageReference, blob, { contentType: "image/png" });
    const imageUrl = await getDownloadURL(imageReference);
    await addDoc(collection(db, "captured_pets"), {
      label: pendingCapture.label,
      confidence: pendingCapture.confidence,
      imageUrl,
      storagePath,
      capturedAt: serverTimestamp(),
      capturedAtEpochMillis: pendingCapture.capturedAtEpochMillis,
      boundingBox: pendingCapture.boundingBox,
      distanceMeters: pendingCapture.distanceMeters,
      riskLevel: pendingCapture.riskLevel,
      cameraSource: pendingCapture.cameraSource,
      source: "web_capture_store",
      backgroundRemoved: pendingCapture.backgroundRemoved,
      backgroundRemovalMethod: pendingCapture.backgroundRemovalMethod,
      trackId: pendingCapture.trackId
    });
    captureStoreStatus.textContent = "Saved to collection.";
    captureModal.hidden = true;
    pendingCapture = null;
  } catch (error) {
    console.error(error);
    capturePreviewNote.textContent = "Save failed. Check Firebase Storage rules.";
    saveCaptureButton.disabled = false;
  } finally {
    saveInFlight = false;
  }
}

async function saveDetections(result) {
  for (const detection of result.detections || []) {
    if (!detection.warningTriggered && detection.confidence < 0.45) continue;
    const signature = `${detection.trackId}:${detection.label}:${Math.round(result.timestamp / 3000)}`;
    if (lastSavedBySignature.has(signature)) continue;
    lastSavedBySignature.set(signature, Date.now());
    try {
      await addDoc(collection(db, "detections"), {
        ...detection,
        source: "web_backend",
        createdAt: serverTimestamp()
      });
      firebaseStatus.textContent = "Saved";
    } catch (error) {
      firebaseStatus.textContent = "Error";
      console.error(error);
    }
  }
}

function listenToRecentDetections() {
  const detectionsQuery = query(collection(db, "detections"), orderBy("createdAt", "desc"), limit(8));
  onSnapshot(detectionsQuery, (snapshot) => {
    historyList.innerHTML = "";
    snapshot.docs.forEach((doc) => {
      const data = doc.data();
      const card = document.createElement("article");
      card.className = "history-card";
      const confidence = typeof data.confidence === "number" ? `${(data.confidence * 100).toFixed(0)}%` : "N/A";
      card.innerHTML = `
        <strong>${data.label || "unknown"} - ${confidence}</strong>
        <p>${formatDistance(data.distanceMeters)} - ${data.riskLevel || "N/A"} - ${data.warningLevel || "NONE"}</p>
        <p>${formatTime(data.detectedAtEpochMillis || data.timestamp)}</p>
      `;
      historyList.appendChild(card);
    });
  }, (error) => {
    firebaseStatus.textContent = "Read error";
    console.error(error);
  });
}

function listenToCapturedPets() {
  const capturesQuery = query(collection(db, "captured_pets"), orderBy("capturedAt", "desc"), limit(24));
  onSnapshot(capturesQuery, (snapshot) => {
    collectionItems = snapshot.docs.map((captureDoc) => ({
      id: captureDoc.id,
      ...captureDoc.data()
    }));
    renderCollection();
  }, (error) => {
    firebaseStatus.textContent = "Collection error";
    console.error(error);
  });
}

function setCollectionFilter(filter) {
  collectionFilter = filter;
  collectionTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.filter === filter);
  });
  renderCollection();
}

function renderCollection() {
  const filtered = collectionItems.filter((item) => collectionFilter === "all" || item.label === collectionFilter);
  collectionList.innerHTML = "";
  if (filtered.length === 0) {
    const empty = document.createElement("article");
    empty.className = "history-card";
    empty.innerHTML = "<strong>No captures yet</strong><p>Dog and cat stickers will appear here.</p>";
    collectionList.appendChild(empty);
    return;
  }

  for (const item of filtered) {
    const card = document.createElement("article");
    card.className = "collection-card";
    const confidence = typeof item.confidence === "number" ? `${(item.confidence * 100).toFixed(0)}%` : "N/A";
    const capturedTime = formatFirestoreTime(item.capturedAt) || formatTime(item.capturedAtEpochMillis);
    card.innerHTML = `
      <img src="${escapeAttribute(item.imageUrl || "")}" alt="${escapeAttribute(item.label || "pet")} capture" loading="lazy" />
      <strong>${escapeHtml(item.label || "unknown")} - ${confidence}</strong>
      <span>${capturedTime}</span>
      <span>${formatDistance(item.distanceMeters)} - ${item.riskLevel || "N/A"}</span>
    `;
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-capture-button";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", () => deleteCapturedPet(item));
    card.appendChild(deleteButton);
    collectionList.appendChild(card);
  }
}

async function deleteCapturedPet(item) {
  if (!item?.id) return;
  try {
    if (item.storagePath) {
      await deleteObject(storageRef(storage, item.storagePath));
    }
    await deleteDoc(doc(db, "captured_pets", item.id));
    captureStoreStatus.textContent = "Capture deleted.";
  } catch (error) {
    console.error(error);
    captureStoreStatus.textContent = "Delete failed.";
  }
}

async function startLocationWatch() {
  if (!("geolocation" in navigator)) return;
  navigator.geolocation.watchPosition(
    (position) => {
      latestPosition = {
        latitude: position.coords.latitude,
        longitude: position.coords.longitude,
        bearingDegrees: typeof position.coords.heading === "number" ? position.coords.heading : null,
        vehicleSpeedKmh: typeof position.coords.speed === "number" ? position.coords.speed * 3.6 : null,
        timestamp: position.timestamp
      };
    },
    () => {},
    { enableHighAccuracy: true, maximumAge: 3000, timeout: 10000 }
  );
}

function freshPosition() {
  if (!latestPosition || Date.now() - latestPosition.timestamp > 8000) return {};
  return latestPosition;
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = Math.max(1, Math.round(rect.width));
  overlay.height = Math.max(1, Math.round(rect.height));
}

function formatDistance(value) {
  return typeof value === "number" ? `${value.toFixed(1)} m` : "N/A";
}

function formatTime(epochMillis) {
  if (!epochMillis) return "time unavailable";
  return new Date(epochMillis).toLocaleString();
}

function formatFirestoreTime(timestamp) {
  if (!timestamp?.toDate) return "";
  return timestamp.toDate().toLocaleString();
}

function setStatus(message) {
  statusPill.textContent = message;
}

function detectDeviceType() {
  const ua = navigator.userAgent || "";
  if (/iPhone/i.test(ua)) return "iPhone";
  if (/iPad/i.test(ua)) return "iPad";
  if (/Android/i.test(ua)) return "Android";
  return "Web";
}

function normalizeApiBaseUrl(url) {
  return String(url || "").trim().replace(/\/+$/, "");
}

function getAnonymousId() {
  const key = "airacareAnonymousId";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const randomId = window.crypto?.randomUUID
    ? window.crypto.randomUUID()
    : `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  const generated = `anon_${randomId}`;
  localStorage.setItem(key, generated);
  return generated;
}

function dataUrlToBlob(dataUrl) {
  return fetch(dataUrl).then((response) => response.blob());
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function capitalize(value) {
  const text = String(value || "");
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Pet";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

function playWarningTone() {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.frequency.value = 880;
    gain.gain.value = 0.06;
    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.18);
  } catch {
    // Browser audio can be blocked until the user interacts; visual warning remains active.
  }
}
