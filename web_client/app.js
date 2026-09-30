import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js";
import {
  getAuth,
  signInAnonymously
} from "https://www.gstatic.com/firebasejs/10.13.2/firebase-auth.js";
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
  serverTimestamp,
  where
} from "https://www.gstatic.com/firebasejs/10.13.2/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyDR_Bt2tltKxSfLSvmlW5mQ0uwxlmafy3w",
  authDomain: "airacare-animal-safety.firebaseapp.com",
  projectId: "airacare-animal-safety",
  messagingSenderId: "934286949654",
  appId: "1:934286949654:android:a0c98c97bd4568e95ee437"
};

const DETECTION_INTERVAL_MS = 2500;
const MAX_CAPTURE_WIDTH = 320;
const DETECTION_TIMEOUT_MS = 60000;
const CAMERA_OPEN_TIMEOUT_MS = 15000;
const VIDEO_PLAY_TIMEOUT_MS = 10000;
const PET_LABELS = new Set(["dog", "cat"]);
const STABLE_PET_FRAMES = 2;
const CAPTURE_COOLDOWN_MS = 2500;
const DEFAULT_API_BASE_URL = "https://airacare-web-backend.onrender.com";
const API_BASE_URL = normalizeApiBaseUrl(
  new URLSearchParams(window.location.search).get("api") || DEFAULT_API_BASE_URL
);

const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore(firebaseApp);
const auth = getAuth(firebaseApp);

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

boot();
console.info("[Airacare] Backend base URL:", API_BASE_URL);

async function boot() {
  checkBackend();
  const authReady = await initializeAnonymousAuth();
  if (!authReady) {
    firebaseStatus.textContent = "Auth required";
    collectionItems = [];
    renderCollection();
    return;
  }
  listenToRecentDetections();
  listenToCapturedPets();
}

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
  let openedStream = null;
  try {
    setStatus("Opening camera");
    startButton.disabled = true;
    await startLocationWatch();
    openedStream = await withTimeout(
      requestCameraStream(),
      CAMERA_OPEN_TIMEOUT_MS,
      "Camera permission timed out. Allow camera access and reload."
    );
    stream = openedStream;
    video.srcObject = stream;
    await withTimeout(waitForVideoReady(), VIDEO_PLAY_TIMEOUT_MS, "Camera video did not become ready.");
    await withTimeout(video.play(), VIDEO_PLAY_TIMEOUT_MS, "Camera playback failed to start.");
    resizeOverlay();
    running = true;
    stopButton.disabled = false;
    setStatus("Detecting");
    detectLoop();
  } catch (error) {
    if (openedStream && openedStream !== stream) openedStream.getTracks().forEach((track) => track.stop());
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    const message = cameraErrorMessage(error);
    console.error("[Airacare] Camera start failed", {
      name: error?.name || null,
      message: error?.message || String(error),
      constraint: error?.constraint || null,
      isSecureContext: window.isSecureContext,
      hasMediaDevices: Boolean(navigator.mediaDevices?.getUserMedia)
    });
    setStatus(message);
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
  return captureCanvas.toDataURL("image/jpeg", 0.72);
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
    .filter(isValidPetDetection)
    .sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0] || null;
}

function updateCaptureStoreButton() {
  const canCapture = Boolean(stablePetCandidate && running && !captureInFlight && hasCurrentVideoFrame());
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
  const petDetection = { ...stablePetCandidate, boundingBox: { ...stablePetCandidate.boundingBox } };
  const validationError = validateCaptureInputs(petDetection, latestDetectionResult);
  if (validationError) {
    captureStoreStatus.textContent = validationError;
    console.warn("[Airacare] Capture blocked:", validationError, petDetection, latestDetectionResult);
    return;
  }
  if (Date.now() - lastCaptureAt < CAPTURE_COOLDOWN_MS) {
    captureStoreStatus.textContent = "Please wait before capturing again.";
    return;
  }

  captureInFlight = true;
  lastCaptureAt = Date.now();
  updateCaptureStoreButton();
  captureStoreStatus.textContent = "Creating sticker...";

  try {
    const cropDataUrl = cropPetFromCurrentFrame(petDetection, latestDetectionResult);
    const endpoint = `${API_BASE_URL}/api/remove-background`;
    console.info("[Airacare] Background removal endpoint:", endpoint);
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: cropDataUrl,
        label: petDetection.label
      })
    });
    const text = await response.text();
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    const backgroundResult = JSON.parse(text);
    if (!backgroundResult.ok || !backgroundResult.image) {
      throw new Error(backgroundResult.error || "Background removal failed");
    }
    if (!isValidDataUrl(backgroundResult.image, "image/png")) {
      throw new Error("Background removal returned an invalid PNG preview");
    }
    const previewBlob = await dataUrlToBlob(backgroundResult.image);
    console.info("[Airacare] Sticker blob:", {
      size: previewBlob.size,
      type: previewBlob.type,
      backgroundRemoved: backgroundResult.backgroundRemoved,
      method: backgroundResult.method
    });
    if (!previewBlob.size || previewBlob.type !== "image/png") {
      throw new Error("Background removal returned an empty or non-PNG image");
    }

    pendingCapture = {
      label: String(petDetection.label || "").toLowerCase(),
      confidence: Number(petDetection.confidence),
      imageDataUrl: backgroundResult.image,
      backgroundRemoved: Boolean(backgroundResult.backgroundRemoved),
      backgroundRemovalMethod: backgroundResult.method || "unknown",
      boundingBox: { ...petDetection.boundingBox },
      distanceMeters: petDetection.distanceMeters ?? null,
      riskLevel: petDetection.riskLevel || null,
      cameraSource: petDetection.cameraSource || "web_camera_backend",
      trackId: petDetection.trackId ?? null,
      capturedAtEpochMillis: Date.now()
    };
    const captureValidationError = validatePendingCapture(pendingCapture);
    if (captureValidationError) {
      pendingCapture = null;
      throw new Error(captureValidationError);
    }
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
  console.info("[Airacare] Capture crop:", {
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    modelFrameWidth: frameWidth,
    modelFrameHeight: frameHeight,
    boundingBox: box,
    crop: { left, top, right, bottom },
    cropWidth,
    cropHeight
  });
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
  const dataUrl = petCropCanvas.toDataURL("image/jpeg", 0.88);
  console.info("[Airacare] Crop data URL chars:", dataUrl.length);
  return dataUrl;
}

function showCapturePreview(capture) {
  const validationError = validatePendingCapture(capture);
  if (validationError) {
    console.warn("[Airacare] Preview blocked:", validationError, capture);
    captureStoreStatus.textContent = validationError;
    saveCaptureButton.disabled = true;
    return;
  }
  captureModalTitle.textContent = `Captured ${capitalize(capture.label)}`;
  capturePreviewImage.src = capture.imageDataUrl;
  capturePreviewLabel.textContent = capture.label;
  capturePreviewConfidence.textContent = `${(capture.confidence * 100).toFixed(0)}%`;
  capturePreviewTime.textContent = formatTime(capture.capturedAtEpochMillis);
  capturePreviewNote.textContent = capture.backgroundRemoved
    ? "Transparent sticker created."
    : "Fallback crop created because background removal was uncertain.";
  saveCaptureButton.disabled = Boolean(validatePendingCapture(capture));
  captureModal.hidden = false;
}

function closeCaptureModal() {
  if (saveInFlight) return;
  captureModal.hidden = true;
  pendingCapture = null;
}

async function savePendingCapture() {
  if (!pendingCapture || saveInFlight) return;
  const validationError = validatePendingCapture(pendingCapture);
  if (validationError) {
    capturePreviewNote.textContent = validationError;
    saveCaptureButton.disabled = true;
    console.warn("[Airacare] Save blocked:", validationError, pendingCapture);
    return;
  }
  saveInFlight = true;
  saveCaptureButton.disabled = true;
  capturePreviewNote.textContent = "Saving to collection...";
  try {
    const blob = await dataUrlToBlob(pendingCapture.imageDataUrl);
    console.info("[Airacare] Backend upload blob:", { size: blob.size, type: blob.type });
    if (!blob.size || blob.type !== "image/png") {
      throw new Error("Invalid sticker image blob");
    }
    const uploadResult = await uploadCapturedPetImage(blob, pendingCapture);
    await addDoc(collection(db, "captured_pets"), {
      ownerId: anonymousId,
      label: pendingCapture.label,
      confidence: pendingCapture.confidence,
      imageUrl: uploadResult.imageUrl,
      imagePublicId: uploadResult.imagePublicId || null,
      storageProvider: uploadResult.storageProvider || "backend",
      capturedAt: serverTimestamp(),
      capturedAtEpochMillis: pendingCapture.capturedAtEpochMillis,
      boundingBox: pendingCapture.boundingBox,
      distanceMeters: pendingCapture.distanceMeters,
      riskLevel: pendingCapture.riskLevel,
      cameraSource: pendingCapture.cameraSource,
      source: "capture_store",
      backgroundRemoved: pendingCapture.backgroundRemoved,
      backgroundRemovalMethod: pendingCapture.backgroundRemovalMethod,
      trackId: pendingCapture.trackId
    });
    captureStoreStatus.textContent = "Saved to collection.";
    captureModal.hidden = true;
    pendingCapture = null;
  } catch (error) {
    logFirebaseError("Save capture failed", error);
    capturePreviewNote.textContent = firebaseUserMessage(error, "Save failed. Check backend image upload or Firestore rules.");
    saveCaptureButton.disabled = false;
  } finally {
    saveInFlight = false;
  }
}

async function uploadCapturedPetImage(blob, capture) {
  const endpoint = `${API_BASE_URL}/api/captured-pets/upload`;
  const formData = new FormData();
  formData.append("image", blob, `${capture.capturedAtEpochMillis}_${capture.label}.png`);
  formData.append("label", capture.label);
  formData.append("ownerId", anonymousId);
  formData.append("capturedAtEpochMillis", String(capture.capturedAtEpochMillis));
  console.info("[Airacare] Captured pet upload endpoint:", endpoint);
  const response = await fetch(endpoint, {
    method: "POST",
    body: formData
  });
  const text = await response.text();
  console.info("[Airacare] Captured pet upload status:", response.status);
  console.info("[Airacare] Captured pet upload response:", text);
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || `Image upload failed with HTTP ${response.status}`);
  }
  if (!response.ok || !data.ok || !data.imageUrl) {
    throw new Error(data.error || `Image upload failed with HTTP ${response.status}`);
  }
  return data;
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
        ownerId: anonymousId,
        source: "web_backend",
        createdAt: serverTimestamp()
      });
      firebaseStatus.textContent = "Saved";
    } catch (error) {
      firebaseStatus.textContent = firebaseUserMessage(error, "Save error");
      logFirebaseError("Detection save failed", error);
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
    logFirebaseError("Detections listener failed", error);
  });
}

function listenToCapturedPets() {
  const capturesQuery = query(
    collection(db, "captured_pets"),
    where("ownerId", "==", anonymousId),
    limit(24)
  );
  onSnapshot(capturesQuery, (snapshot) => {
    collectionItems = snapshot.docs.map((captureDoc) => ({
      id: captureDoc.id,
      ...captureDoc.data()
    })).sort((a, b) => Number(b.capturedAtEpochMillis || 0) - Number(a.capturedAtEpochMillis || 0));
    renderCollection();
  }, (error) => {
    firebaseStatus.textContent = firebaseUserMessage(error, "Collection error");
    logFirebaseError("Captured pets listener failed", error);
    collectionItems = [];
    renderCollection();
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
    if (item.imagePublicId) {
      await deleteCapturedPetImage(item.imagePublicId);
    }
    await deleteDoc(doc(db, "captured_pets", item.id));
    captureStoreStatus.textContent = "Capture deleted.";
  } catch (error) {
    logFirebaseError("Delete capture failed", error);
    captureStoreStatus.textContent = firebaseUserMessage(error, "Delete failed.");
  }
}

async function deleteCapturedPetImage(imagePublicId) {
  const endpoint = `${API_BASE_URL}/api/captured-pets/delete`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imagePublicId })
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(text || `Image delete failed with HTTP ${response.status}`);
  }
  if (!response.ok || !data.ok) {
    throw new Error(data.error || `Image delete failed with HTTP ${response.status}`);
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

async function requestCameraStream() {
  if (!window.isSecureContext) {
    throw new Error("Camera needs HTTPS. Open the Firebase Hosting link, not plain HTTP.");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser does not support camera access.");
  }

  const preferred = {
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: false
  };

  try {
    return await navigator.mediaDevices.getUserMedia(preferred);
  } catch (error) {
    console.warn("[Airacare] Preferred camera failed, retrying default camera", {
      name: error?.name || null,
      message: error?.message || String(error),
      constraint: error?.constraint || null
    });
    if (["NotAllowedError", "SecurityError"].includes(error?.name)) throw error;
    return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
}

function waitForVideoReady() {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth && video.videoHeight) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", handleReady);
      video.removeEventListener("canplay", handleReady);
      video.removeEventListener("error", handleError);
    };
    const handleReady = () => {
      if (video.videoWidth && video.videoHeight) {
        cleanup();
        resolve();
      }
    };
    const handleError = () => {
      cleanup();
      reject(new Error("Camera video element failed to load."));
    };
    video.addEventListener("loadedmetadata", handleReady);
    video.addEventListener("canplay", handleReady);
    video.addEventListener("error", handleError);
  });
}

async function initializeAnonymousAuth() {
  try {
    const credential = await signInAnonymously(auth);
    if (credential.user?.uid) {
      anonymousId = credential.user.uid;
      console.info("[Airacare] Firebase anonymous auth ready:", anonymousId);
      firebaseStatus.textContent = "Ready";
      return true;
    }
    firebaseStatus.textContent = "Auth failed";
    return false;
  } catch (error) {
    logFirebaseError("Anonymous auth failed", error);
    firebaseStatus.textContent = firebaseUserMessage(error, "Firebase auth failed");
    return false;
  }
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

function withTimeout(promise, timeoutMs, timeoutMessage) {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timeout);
        reject(error);
      }
    );
  });
}

function cameraErrorMessage(error) {
  const name = String(error?.name || "");
  const message = String(error?.message || "");
  if (!window.isSecureContext) return "Camera needs HTTPS";
  if (name === "NotAllowedError" || name === "SecurityError") return "Camera permission denied";
  if (name === "NotFoundError" || name === "DevicesNotFoundError") return "No camera found";
  if (name === "NotReadableError" || name === "TrackStartError") return "Camera is already in use";
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") return "Requested camera is unavailable";
  if (message) return message;
  return "Camera failed";
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

function isValidPetDetection(detection) {
  const label = String(detection?.label || "").toLowerCase();
  if (!PET_LABELS.has(label)) return false;
  if (typeof detection.confidence !== "number" || !Number.isFinite(detection.confidence)) return false;
  return isValidBoundingBox(detection.boundingBox);
}

function isValidBoundingBox(box) {
  if (!box) return false;
  const values = [box.left, box.top, box.right, box.bottom].map(Number);
  return values.every(Number.isFinite) && values[2] > values[0] && values[3] > values[1];
}

function hasCurrentVideoFrame() {
  return Boolean(video.videoWidth > 0 && video.videoHeight > 0 && !video.paused && !video.ended);
}

function validateCaptureInputs(detection, result) {
  if (!running) return "Start the camera before capturing.";
  if (!hasCurrentVideoFrame()) return "Camera frame is not ready yet.";
  if (!result || typeof result.frameWidth !== "number" || typeof result.frameHeight !== "number") {
    return "Detection frame data is unavailable.";
  }
  if (!isValidPetDetection(detection)) {
    return "Capture is available only for a valid dog or cat detection.";
  }
  return "";
}

function validatePendingCapture(capture) {
  if (!capture) return "No capture is ready.";
  if (!PET_LABELS.has(String(capture.label || "").toLowerCase())) return "Capture label is invalid.";
  if (typeof capture.confidence !== "number" || !Number.isFinite(capture.confidence)) return "Capture confidence is unavailable.";
  if (!isValidBoundingBox(capture.boundingBox)) return "Capture bounding box is invalid.";
  if (!capture.capturedAtEpochMillis) return "Capture time is unavailable.";
  if (!isValidDataUrl(capture.imageDataUrl, "image/png")) return "Sticker image is unavailable.";
  return "";
}

function isValidDataUrl(value, mimeType) {
  return typeof value === "string" && value.startsWith(`data:${mimeType};base64,`) && value.length > `data:${mimeType};base64,`.length;
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

function logFirebaseError(context, error) {
  console.error(`[Airacare] ${context}`, {
    code: error?.code || null,
    message: error?.message || String(error),
    name: error?.name || null,
    stack: error?.stack || null
  });
}

function firebaseUserMessage(error, fallback) {
  const code = String(error?.code || "");
  const message = String(error?.message || "");
  if (code.includes("auth/operation-not-allowed") || code.includes("auth/admin-restricted-operation")) {
    return "Enable Firebase Anonymous Auth.";
  }
  if (code.includes("auth/") || message.includes("CONFIGURATION_NOT_FOUND")) {
    return "Firebase authentication failed.";
  }
  if (code.includes("permission-denied") || code.includes("unauthorized")) {
    return "Firebase rules blocked access.";
  }
  return fallback;
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
