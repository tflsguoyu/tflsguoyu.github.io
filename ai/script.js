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
let lastFocusedElement = null;

function renderLinks(links = []) {
  if (!links.length) return "";
  const renderedLinks = links
    .map(([label, href]) => `<a href="${href}" target="${href.startsWith("mailto:") ? "_self" : "_blank"}" rel="noreferrer">${label}</a>`)
    .join("");
  return `<div class="link-row">${renderedLinks}</div>`;
}

function openCard(name, trigger) {
  const card = paperCards[name];
  if (!card) return;

  lastFocusedElement = trigger || document.activeElement;
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
  if (lastFocusedElement) lastFocusedElement.focus();
}

document.addEventListener("click", (event) => {
  const hotspot = event.target.closest("[data-card]");
  if (hotspot) {
    openCard(hotspot.dataset.card, hotspot);
    return;
  }

  if (event.target.closest("[data-close-modal]")) {
    closeCard();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !modal.hidden) {
    closeCard();
  }
});
