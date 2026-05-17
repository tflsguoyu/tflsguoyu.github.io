const scholarUrl = "https://scholar.google.com/citations?user=V2BnBGIAAAAJ&hl=en";

const paperCards = {
  path: {
    kicker: "Wall Map",
    title: "Academic Journey",
    intro:
      "The map traces the route behind the room: education, research appointments, internships, and industry research.",
    html: `
      <ol>
        <li><strong>Changsha</strong> - B.S. in Mathematics, Central South University.</li>
        <li><strong>Beijing / Shenzhen</strong> - M.S. in Computer Science, Chinese Academy of Sciences / SIAT.</li>
        <li><strong>Singapore</strong> - Research Associate, Nanyang Technological University and BeingThere Centre.</li>
        <li><strong>Irvine</strong> - Ph.D. in Computer Science, University of California, Irvine.</li>
        <li><strong>Industry research and internships</strong> - Autodesk, Megvii, Adobe, Facebook, and Tencent Pixel Lab.</li>
      </ol>
    `,
    links: [
      ["Google Scholar", scholarUrl],
      ["UC Irvine", "https://www.uci.edu/"],
      ["Shuang Zhao", "https://shuangz.com/"],
    ],
  },
  contact: {
    kicker: "Pinned Note",
    title: "Contact",
    intro: "A small note on the wall for the practical exits from the room.",
    links: [
      ["Email", "mailto:tflsguoyu@gmail.com"],
      ["Google Scholar", scholarUrl],
      ["GitHub", "https://github.com/tflsguoyu"],
    ],
  },
  materialgan: {
    kicker: "Solid Wood Desktop",
    title: "MaterialGAN: Reflectance Capture using a Generative SVBRDF Model",
    meta: "Yu Guo, Cameron Smith, Milos Hasan, Kalyan Sunkavalli, Shuang Zhao. ACM Transactions on Graphics, 2020.",
    intro:
      "The wooden desktop opens the material-capture work: reconstructing spatially varying BRDFs from limited photographs using a generative SVBRDF prior.",
    links: [
      ["Paper", "https://github.com/tflsguoyu/materialgan_paper/blob/master/materialgan.pdf"],
      ["Project and code", "https://github.com/tflsguoyu/svbrdf-diff-renderer"],
      ["Scholar", scholarUrl],
    ],
  },
  "layered-bsdf": {
    kicker: "Cyber Helmet",
    title: "Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs",
    meta: "Yu Guo, Milos Hasan, Shuang Zhao. ACM Transactions on Graphics, 2018.",
    intro:
      "The helmet maps to layered appearance: an unbiased Monte Carlo model for arbitrary layered BSDFs with surface and volumetric scattering.",
    links: [
      ["Paper", "https://github.com/tflsguoyu/layeredbsdf_paper/blob/master/layeredbsdf.pdf"],
      ["Project and code", "https://github.com/tflsguoyu/layeredbsdf/"],
      ["Scholar", scholarUrl],
    ],
  },
  bayesian: {
    kicker: "Textured Wallpaper",
    title: "A Bayesian Inference Framework for Procedural Material Parameter Estimation",
    meta: "Yu Guo, Milos Hasan, Lingqi Yan, Shuang Zhao. Computer Graphics Forum, 2020.",
    intro:
      "The patterned wall opens procedural material estimation: fitting editable material models from photographs with optimization and Bayesian inference.",
    links: [
      ["Paper", "https://github.com/tflsguoyu/proceduralmat_paper/blob/master/proceduralmat.pdf"],
      ["Project and code", "https://github.com/tflsguoyu/proceduralmat/"],
      ["Scholar", scholarUrl],
    ],
  },
  faces: {
    kicker: "Bust Sculpture",
    title: "3D Faces are Recognized More Accurately and Faster than 2D Faces, but with Similar Inversion Effects",
    meta: "Derric Eng, Belle Yick, Yu Guo, Hong Xu, Miriam Reiner, TJ Cham, SH Chen. Vision Research, 2017.",
    intro:
      "The bust sculpture points to earlier vision work comparing recognition performance for 3D and 2D faces under upright and inverted presentation.",
    links: [
      ["Scholar", scholarUrl],
    ],
  },
  tryon: {
    kicker: "Blue-Light Glasses",
    title: "A Virtual Try-on System for Prescription Eyeglasses",
    meta: "Qian Zhang, Yu Guo, Pierre-Yves Laffont, Tobias Martin, Markus Gross. IEEE Computer Graphics and Applications, 2017.",
    intro:
      "The glasses open a virtual try-on system that models corrective-lens distortion, reflections, and shading for a more realistic mirror-like result.",
    links: [
      ["Video", "https://youtu.be/_fckwZCzCgc"],
      ["Scholar", scholarUrl],
    ],
  },
  fabric: {
    kicker: "Study Curtains",
    title: "Fiber-Level Woven Fabric Capture from a Single Photo",
    meta: "Woven fabric appearance capture work listed on Google Scholar.",
    intro:
      "The semi-translucent curtains map to fabric capture: recovering detailed woven-fiber appearance from limited image input.",
    links: [
      ["Project", "https://wangningbei.github.io/2022/Fabrics.html"],
      ["Scholar", scholarUrl],
    ],
  },
  "beyond-mie": {
    kicker: "Frosted Glass Lamp",
    title: "Beyond Mie Theory: Systematic Computation of Bulk Scattering Parameters based on Microphysical Wave Optics",
    meta: "Yu Guo, Adrian Jarabo, Shuang Zhao. ACM Transactions on Graphics, 2021.",
    intro:
      "The glowing frosted lamp opens the scattering work: computing bulk scattering parameters beyond the far-field assumptions of Lorenz-Mie theory.",
    links: [
      ["Paper", "https://github.com/tflsguoyu/beyondmie_paper/blob/master/beyondmie.pdf"],
      ["Project and code", "https://github.com/tflsguoyu/beyondmie"],
      ["Scholar", scholarUrl],
    ],
  },
  bigs: {
    kicker: "3D Printed Ornament",
    title: "BiGS: Bidirectional Gaussian Primitives for Relightable 3D Gaussian Splatting",
    meta: "Relightable 3D Gaussian splatting work listed on Google Scholar.",
    intro:
      "The 3D printed ornament opens BiGS, a relightable 3D Gaussian representation designed for more expressive appearance and lighting control.",
    links: [
      ["Project", "https://desmondlzy.me/publications/bigs/"],
      ["arXiv", "https://arxiv.org/abs/2408.13370"],
      ["Scholar", scholarUrl],
    ],
  },
  deformable: {
    kicker: "Squishy Toy",
    title: "Textureless Deformable Object Tracking with Invisible Markers",
    meta: "Textureless deformable tracking work listed on Google Scholar.",
    intro:
      "The squishy toy maps to deformable tracking: estimating motion and deformation for objects that lack visible texture cues.",
    links: [
      ["Project", "https://fluorescentdot.github.io"],
      ["Scholar", scholarUrl],
    ],
  },
  "video-editing": {
    kicker: "Monitor",
    title: "Physically Based Video Editing",
    meta: "Jean-Charles Bazin, Claudia Pluss, Yu Guo, Tobias Martin, Alec Jacobson, Markus Gross. Computer Graphics Forum, 2016.",
    intro:
      "The monitor opens video editing work that combines image-aware editing with physically based simulation to produce plausible video manipulations.",
    links: [
      ["Video", "https://youtu.be/bBzmlCU5FEo"],
      ["Scholar", scholarUrl],
    ],
  },
  epbr: {
    kicker: "Glass Trophy",
    title: "ePBR: Extended PBR Materials in Image Synthesis",
    meta: "Yu Guo, Zhiqiang Lao, Xiyun Song, Yubin Zhou, Zongfang Lin, Heather Yu. CVPR Workshops, 2025.",
    intro:
      "The transparent trophy opens ePBR, which extends PBR-style image decomposition with reflection and transmission for materials such as glass and windows.",
    links: [
      ["CVF", "https://openaccess.thecvf.com/content/CVPR2025W/CV4Metaverse/html/Guo_ePBR_Extended_PBR_Materials_in_Image_Synthesis_CVPRW_2025_paper.html"],
      ["arXiv", "https://arxiv.org/abs/2504.17062"],
      ["Scholar", scholarUrl],
    ],
  },
  sirr: {
    kicker: "Reflective Photo Frame",
    title: "SIRR-LMM: Single-image Reflection Removal via Large Multimodal Model",
    meta: "Yu Guo, Zhiqiang Lao, Xiyun Song, Yubin Zhou, Heather Yu. WACV Workshops, 2026.",
    intro:
      "The reflective frame glass opens SIRR-LMM, a single-image reflection removal approach using synthetic reflection scenarios and a large multimodal model.",
    links: [
      ["CVF", "https://openaccess.thecvf.com/content/WACV2026W/GAIP/html/Guo_SIRR-LMM_Single-image_reflection_removal_via_Large_Multimodal_Model_WACVW_2026_paper.html"],
      ["arXiv", "https://arxiv.org/abs/2601.07209"],
      ["Scholar", scholarUrl],
    ],
  },
  "grain-sand": {
    kicker: "Anime Figure",
    title: "Seeing A 3D World in A Grain of Sand",
    meta: "3D reconstruction / perception work listed on Google Scholar.",
    intro:
      "The figurine opens the grain-of-sand project, a newer work on seeing and recovering 3D structure from tiny visual evidence.",
    links: [
      ["arXiv", "https://arxiv.org/abs/2503.00260"],
      ["Scholar", scholarUrl],
    ],
  },
};

const modal = document.querySelector("[data-modal]");
const modalKicker = document.querySelector("[data-modal-kicker]");
const modalTitle = document.querySelector("[data-modal-title]");
const modalBody = document.querySelector("[data-modal-body]");
const roomCanvas = document.querySelector(".room-canvas");
const maskTooltip = document.querySelector("[data-mask-tooltip]");
let lastFocusedElement = null;

const maskHotspots = [
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

const loadedMasks = new Map();
const MASK_THRESHOLD = 127;
let activeMaskCard = null;
let rectHotspotFallback = false;
let pendingHoverPoint = null;
let hoverFrame = null;

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
  if (!links.length) return "";
  const renderedLinks = links
    .map(([label, href]) => `<a href="${href}" target="${href.startsWith("mailto:") ? "_self" : "_blank"}" rel="noreferrer">${label}</a>`)
    .join("");
  return `<div class="link-row">${renderedLinks}</div>`;
}

function loadMask(entry) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        const data = new Uint8Array(canvas.width * canvas.height);
        const bounds = {
          minX: canvas.width,
          minY: canvas.height,
          maxX: -1,
          maxY: -1,
        };

        for (let i = 0, j = 0; i < data.length; i += 1, j += 4) {
          const value = Math.max(pixels[j], pixels[j + 1], pixels[j + 2]);
          if (value <= MASK_THRESHOLD) continue;

          const x = i % canvas.width;
          const y = Math.floor(i / canvas.width);
          data[i] = 1;
          if (x < bounds.minX) bounds.minX = x;
          if (y < bounds.minY) bounds.minY = y;
          if (x > bounds.maxX) bounds.maxX = x;
          if (y > bounds.maxY) bounds.maxY = y;
        }

        if (bounds.maxX < 0) {
          resolve(false);
          return;
        }

        loadedMasks.set(entry.card, {
          width: canvas.width,
          height: canvas.height,
          data,
          bounds,
        });
        resolve(true);
      } catch {
        resolve(false);
      }
    };
    image.onerror = () => resolve(false);
    image.src = entry.mask;
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
  if (!point) return null;

  for (const entry of maskHotspots) {
    const mask = loadedMasks.get(entry.card);
    if (!mask) continue;
    const px = Math.min(mask.width - 1, Math.max(0, Math.floor(point.x * mask.width)));
    const py = Math.min(mask.height - 1, Math.max(0, Math.floor(point.y * mask.height)));
    const { bounds } = mask;
    if (px < bounds.minX || px > bounds.maxX || py < bounds.minY || py > bounds.maxY) continue;
    if (mask.data[py * mask.width + px]) return { ...entry, point };
  }

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
  const card = paperCards[name];
  if (!card) return;

  lastFocusedElement = trigger || null;
  modalKicker.textContent = card.kicker;
  modalTitle.textContent = card.title;
  modalBody.innerHTML = `
    ${card.meta ? `<p class="paper-meta">${card.meta}</p>` : ""}
    <p>${card.intro}</p>
    ${card.html || ""}
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
Promise.all(maskHotspots.map(loadMask)).then((results) => {
  if (results.some((loaded) => !loaded)) {
    enableRectHotspotFallback();
  }
});
