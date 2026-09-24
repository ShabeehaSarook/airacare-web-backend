import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js";
import {
  addDoc,
  collection,
  getFirestore,
  limit,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.13.2/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyDR_Bt2tltKxSfLSvmlW5mQ0uwxlmafy3w",
  authDomain: "airacare-animal-safety.firebaseapp.com",
  projectId: "airacare-animal-safety",
  storageBucket: "airacare-animal-safety.firebasestorage.app",
  messagingSenderId: "934286949654",
  appId: "1:934286949654:android:a0c98c97bd4568e95ee437"
};

const DETECTION_INTERVAL_MS = 700;
const MAX_CAPTURE_WIDTH = 640;
const DEFAULT_API_BASE_URL = "https://airacare-web-backend.onrender.com";
const API_BASE_URL = normalizeApiBaseUrl(
  new URLSearchParams(window.location.search).get("api") || DEFAULT_API_BASE_URL
);

const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore(firebaseApp);

const video = document.getElementById("cameraVideo");
const overlay = document.getElementById("overlayCanvas");
const overlayCtx = overlay.getContext("2d");
const captureCanvas = document.getElementById("captureCanvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const statusPill = document.getElementById("statusPill");
const warningBanner = document.getElementById("warningBanner");
const latestLabel = document.getElementById("latestLabel");
const latestDistance = document.getElementById("latestDistance");
const latestRisk = document.getElementById("latestRisk");
const firebaseStatus = document.getElementById("firebaseStatus");
const modelStatus = document.getElementById("modelStatus");
const historyList = document.getElementById("historyList");

let stream = null;
let running = false;
let loopTimer = null;
let latestPosition = null;
let lastSavedBySignature = new Map();

startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
window.addEventListener("resize", resizeOverlay);

listenToRecentDetections();
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
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Stopped");
}

async function detectLoop() {
  if (!running) return;
  try {
    const image = captureFrame();
    const endpoint = `${API_BASE_URL}/api/detect`;
    console.info("[Airacare] Detection endpoint:", endpoint);
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image,
        ...freshPosition(),
        deviceType: detectDeviceType()
      })
    });
    const text = await response.text();
    console.info("[Airacare] Detection status:", response.status);
    console.info("[Airacare] Detection response:", text);
    if (!response.ok) throw new Error(text || `HTTP ${response.status}`);
    const result = JSON.parse(text);
    drawDetections(result);
    updateLatest(result);
    await saveDetections(result);
  } catch (error) {
    console.error(error);
    setStatus("Detection error");
  } finally {
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
