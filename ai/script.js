const CONTENT_URL = "assets/content.json";
let contentCards = {};

function escapeHTML(value = "") {
  return String(value).replace(/[&<>"]/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
  })[character]);
}

function renderParagraphs(paragraphs = []) {
  return paragraphs.map((paragraph) => `<p>${escapeHTML(paragraph)}</p>`).join("");
}

function renderDetails(details = []) {
  if (!details.length) return "";
  const items = details.map((item) => `<li>${escapeHTML(item)}</li>`).join("");
  return `<ol>${items}</ol>`;
}

function normalizeLinks(links = []) {
  return links.map((link) => Array.isArray(link) ? { label: link[0], href: link[1] } : link);
}

async function loadContent() {
  const response = await fetch(CONTENT_URL);
  if (!response.ok) throw new Error(`Unable to load ${CONTENT_URL}`);
  contentCards = await response.json();
  maskHotspots.forEach((entry) => {
    entry.label = contentCards[entry.card]?.hover || entry.label;
    if (entry.button && contentCards[entry.card]?.hover) {
      entry.button.setAttribute("aria-label", contentCards[entry.card].hover);
    }
  });
}

const modal = document.querySelector("[data-modal]");
const modalTitle = document.querySelector("[data-modal-title]");
const modalBody = document.querySelector("[data-modal-body]");
const roomPage = document.querySelector(".room-page");
const roomCanvas = document.querySelector(".room-canvas");
const maskTooltip = document.querySelector("[data-mask-tooltip]");
let lastFocusedElement = null;

const maskHotspots = [
  ["profile", "Yu Guo profile"],
  ["contact", "Contact and external links"],
  ["sirr", "SIRR-LMM - reflective frame glass"],
  ["faces", "3D faces vs 2D faces - bust"],
  ["beyond-mie", "Beyond Mie Theory - frosted glass lamp"],
  ["bigs", "BiGS - 3D printed ornament"],
  ["tryon", "Virtual try-on - blue-light glasses"],
  ["deformable", "Textureless deformable object - squishy toy"],
  ["video-editing", "Physically based video editing - monitor"],
  ["epbr", "ePBR - glass trophy"],
  ["grain-sand", "Seeing a 3D world in a grain of sand - figure"],
  ["layered-bsdf", "Layered BSDF - cyber helmet"],
  ["fabric", "Woven fabric capture - curtains"],
  ["path", "Education, internships, and work path"],
  ["bayesian", "Bayesian materials - textured wallpaper"],
  ["materialgan", "MaterialGAN - wooden desktop"],
].map(([card, label, mask]) => ({
  card,
  label,
  mask: mask || `assets/masks/${card}.png`,
  highlightMask: `assets/highlights/${card}.png`,
  button: document.querySelector(`[data-card="${card}"]`),
  highlight: null,
}));

const HIT_MAP_URL = "assets/hit-map.png";
let activeMaskCard = null;
let rectHotspotFallback = false;
let pendingHoverPoint = null;
let hoverFrame = null;
let hitMap = null;

function createMaskHighlights() {
  maskHotspots.forEach((entry) => {
    const image = document.createElement("img");
    image.className = "mask-highlight";
    image.dataset.maskHighlight = entry.card;
    image.src = entry.highlightMask;
    image.alt = "";
    roomCanvas.insertBefore(image, maskTooltip);
    entry.highlight = image;
  });
}

function renderLinks(links = []) {
  const normalizedLinks = normalizeLinks(links).filter((link) => link?.label && link?.href);
  if (!normalizedLinks.length) return "";
  const renderedLinks = normalizedLinks
    .map((link) => {
      const href = String(link.href);
      const target = href.startsWith("mailto:") ? "_self" : "_blank";
      return `<a href="${escapeHTML(href)}" target="${target}" rel="noreferrer">${escapeHTML(link.label)}</a>`;
    })
    .join("");
  return `<div class="link-row">${renderedLinks}</div>`;
}

function loadHitMap() {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(image, 0, 0);
        hitMap = {
          width: canvas.width,
          height: canvas.height,
          data: context.getImageData(0, 0, canvas.width, canvas.height).data,
        };
        resolve(true);
      } catch {
        resolve(false);
      }
    };
    image.onerror = () => resolve(false);
    image.src = HIT_MAP_URL;
  });
}

function enableRectHotspotFallback() {
  rectHotspotFallback = true;
  roomCanvas.classList.add("uses-rect-hotspots");
  setMaskHover(null);
}

function getCanvasPoint(event) {
  const rect = roomCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { x, y };
}

function hitTestMask(event) {
  const point = getCanvasPoint(event);
  if (!point || !hitMap) return null;

  const px = Math.min(hitMap.width - 1, Math.max(0, Math.floor(point.x * hitMap.width)));
  const py = Math.min(hitMap.height - 1, Math.max(0, Math.floor(point.y * hitMap.height)));
  const index = hitMap.data[(py * hitMap.width + px) * 4];
  const entry = maskHotspots[index - 1];
  if (entry) return { ...entry, point };

  return null;
}

function scheduleMaskHover(event) {
  pendingHoverPoint = {
    clientX: event.clientX,
    clientY: event.clientY,
  };

  if (hoverFrame) return;
  hoverFrame = requestAnimationFrame(() => {
    hoverFrame = null;
    if (!pendingHoverPoint || !modal.hidden) return;
    setMaskHover(hitTestMask(pendingHoverPoint));
    pendingHoverPoint = null;
  });
}

function centerMobileRoomView() {
  const isMobilePortrait = window.matchMedia("(max-width: 720px) and (orientation: portrait)").matches;
  if (!isMobilePortrait || !roomPage || !roomCanvas) return;

  const maxScroll = roomCanvas.scrollWidth - roomPage.clientWidth;
  if (maxScroll <= 0) return;
  roomPage.scrollLeft = maxScroll * 0.38;
}

function setMaskHover(hit) {
  const nextCard = hit?.card || null;
  if (activeMaskCard !== nextCard) {
    maskHotspots.forEach((entry) => entry.highlight?.classList.remove("is-active"));
    if (hit?.highlight) hit.highlight.classList.add("is-active");
    activeMaskCard = nextCard;
  }

  roomCanvas.classList.toggle("is-mask-hover", Boolean(hit));
  if (!hit) {
    maskTooltip.hidden = true;
    return;
  }

  maskTooltip.textContent = hit.label;
  maskTooltip.style.left = `${hit.point.x * 100}%`;
  maskTooltip.style.top = `${hit.point.y * 100}%`;
  maskTooltip.hidden = false;
}

function openCard(name, trigger) {
  const card = contentCards[name];
  if (!card) return;

  lastFocusedElement = trigger || null;
  modalTitle.textContent = card.title || card.object || name;
  modalBody.innerHTML = `
    ${card.meta ? `<p class="paper-meta">${escapeHTML(card.meta)}</p>` : ""}
    ${renderParagraphs(card.content)}
    ${renderDetails(card.details)}
    ${renderLinks(card.links)}
  `;
  modal.hidden = false;
  document.body.style.overflow = "hidden";
  document.querySelector(".modal-close").focus();
}

function closeCard() {
  modal.hidden = true;
  document.body.style.overflow = "";
  setMaskHover(null);
  if (lastFocusedElement) lastFocusedElement.focus();
}

document.addEventListener("click", (event) => {
  if (event.target.closest("[data-close-modal]")) {
    closeCard();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !modal.hidden) {
    closeCard();
  }
});

roomCanvas.addEventListener("mousemove", (event) => {
  if (!modal.hidden) return;
  scheduleMaskHover(event);
});

roomCanvas.addEventListener("mouseleave", () => {
  if (hoverFrame) cancelAnimationFrame(hoverFrame);
  hoverFrame = null;
  pendingHoverPoint = null;
  setMaskHover(null);
});

roomCanvas.addEventListener("click", (event) => {
  if (rectHotspotFallback) return;
  if (!modal.hidden) return;
  const hit = hitTestMask(event);
  if (hit) openCard(hit.card);
});

maskHotspots.forEach((entry) => {
  entry.button?.addEventListener("click", (event) => {
    if (!rectHotspotFallback) return;
    event.stopPropagation();
    if (!modal.hidden) return;
    openCard(entry.card, entry.button);
  });
});

createMaskHighlights();
Promise.all([loadContent(), loadHitMap()])
  .then(([, hitMapLoaded]) => {
    if (!hitMapLoaded) enableRectHotspotFallback();
  })
  .catch((error) => {
    console.error(error);
    enableRectHotspotFallback();
  });
window.addEventListener("load", centerMobileRoomView, { once: true });
