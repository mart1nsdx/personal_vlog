// ── DATA ────────────────────────────────────────────────────────────────────

const WHAT_I_DO = [
  {
    icon: "⚡",
    title: "Software Development",
    text: "I like building things that actually work. I move across the stack as needed and focus on shipping, not on making it perfect before anyone sees it.",
    tags: ["Full-Stack", "Product", "Shipping"],
  },
  {
    icon: "🤖",
    title: "AI & Automation",
    text: "I use AI and automation to get things done faster and smarter. I'm genuinely curious about what becomes possible when you put these tools in the right context.",
    tags: ["AI", "LLMs", "Automation", "Tooling"],
  },
  {
    icon: "🚀",
    title: "Entrepreneurship",
    text: "I care about finding real problems and building something around them. The early stage is where I feel most alive. Messy, uncertain, exciting.",
    tags: ["Startups", "Founder", "Strategy"],
  },
  {
    icon: "🌐",
    title: "Startup & Business Club · Uniandes",
    text: "I'm part of the Startup and Business Club at Universidad de los Andes. It's where I connect with other people building things and stay close to the entrepreneurship scene in LATAM.",
    tags: ["Uniandes", "Startups", "Community", "Entrepreneurship"],
  },
  {
    icon: "🛸",
    title: "Student Leader · Spacetech AESS IEEE Uniandes",
    text: "I lead the AESS chapter at Universidad de los Andes, the IEEE branch for Colombia. We get students together around aerospace engineering, orbital mechanics and rocket science. It's one of those groups where people actually geek out.",
    tags: ["IEEE", "AESS", "Spacetech", "Aerospace", "Uniandes", "Leadership"],
  },
];

const PROJECTS = [
  {
    num: "01",
    name: "Personal Vlog · Martin Ardila",
    desc: "This site. Built to learn HTML and JavaScript hands-on while working toward my own portfolio. I also use it to document what I'm working on, what I'm learning and where I'm heading.",
    status: "active",
    statusLabel: "Live",
  },
  {
    num: "02",
    name: "Startup Project [TBA]",
    desc: "Something I'm working on. More details soon. In the meantime I'm also open to internships or roles at startups where I can actually contribute and grow.",
    status: "building",
    statusLabel: "Building",
  },
  {
    num: "03",
    name: "Spacetech IEEE Leadership",
    desc: "Running the aerospace chapter at Uniandes. We study orbital mechanics, rocket science and anything space related. It's more fun than it sounds.",
    status: "active",
    statusLabel: "Active",
  },
  {
    num: "04",
    name: "Preicfes Orion",
    desc: "A prep course business for Colombian students from low income families who want to get into university. Quality education shouldn't depend on your zip code.",
    status: "active",
    statusLabel: "Running",
  },
];

const FUN_PROJECTS = [
  {
    emoji: "💨",
    name: "Hovercraft · Object-Detecting Aerodeslizador",
    desc: "Built a hovercraft from scratch. It hovers, it moves fast and it detects whatever is in front of it using sensors.",
    image: "hovercraft.jpeg",
    video: "hovercraft_clean.mp4",
    tech: [],
  },
];

const FUTURE_GOALS = [
  {
    icon: "🌎",
    title: "Developing ideas that drive real growth for LATAM",
    desc: "I want to build things that actually matter for Latin America. Hard tech, IoT, the stuff that's been missing. Not imported solutions adapted for the region but things built here from scratch, for here first.",
    horizon: "Long-term vision",
  },
];

const CONTACT_LINKS = [
  {
    icon: "💼",
    label: "LinkedIn",
    sub: "martin-ardila-5bb02436b",
    href: "https://www.linkedin.com/in/martin-ardila-5bb02436b",
  },
];

// ── RENDER ───────────────────────────────────────────────────────────────────

function renderWhatIDo() {
  const grid = document.getElementById("what-i-do-cards");
  if (!grid) return;
  grid.innerHTML = WHAT_I_DO.map(
    ({ icon, title, text, tags }) => `
    <div class="card">
      <div class="card-icon">${icon}</div>
      <div class="card-title">${title}</div>
      <div class="card-text">${text}</div>
      <div class="card-tags">${tags.map(t => `<span class="tag">${t}</span>`).join("")}</div>
    </div>`
  ).join("");
}

function renderProjects() {
  const list = document.getElementById("projects-list");
  if (!list) return;
  list.innerHTML = PROJECTS.map(
    ({ num, name, desc, status, statusLabel }) => `
    <div class="project-item">
      <div class="project-num">${num}</div>
      <div class="project-content">
        <div class="project-name">${name}</div>
        <div class="project-desc">${desc}</div>
      </div>
      <div class="project-meta">
        <span class="project-status status-${status}">${statusLabel}</span>
      </div>
    </div>`
  ).join("");
}

function renderFunProjects() {
  const grid = document.getElementById("fun-projects-grid");
  if (!grid) return;
  grid.innerHTML = FUN_PROJECTS.map(
    ({ emoji, name, desc, image, video, tech }) => `
    <div class="fun-card">
      <div class="fun-body">
        <div class="fun-name">${emoji} ${name}</div>
        <div class="fun-desc">${desc}</div>
        <div class="card-tags">${tech.map(t => `<span class="tag">${t}</span>`).join("")}</div>
      </div>
      <div class="fun-media">
        <div class="fun-media-box">
          <img src="${image}" alt="${name}" class="fun-img" />
        </div>
        <div class="fun-media-box">
          ${video
            ? `<video controls class="fun-video"><source src="${video}" /></video>`
            : `<div class="fun-video-placeholder"><span class="vp-icon">▶</span><span>Video coming soon</span></div>`
          }
        </div>
      </div>
    </div>`
  ).join("");
}

function renderFuture() {
  const grid = document.getElementById("future-grid");
  if (!grid) return;
  grid.innerHTML = FUTURE_GOALS.map(
    ({ icon, title, desc, horizon }) => `
    <div class="future-item">
      <div class="future-icon">${icon}</div>
      <div class="future-text">
        <div class="title">${title}</div>
        <div class="desc">${desc}</div>
        <div class="future-horizon">⟶ ${horizon}</div>
      </div>
    </div>`
  ).join("");
}

function renderContact() {
  const wrap = document.getElementById("contact-links");
  if (!wrap) return;
  wrap.innerHTML = CONTACT_LINKS.map(
    ({ icon, label, sub, href }) => `
    <a href="${href}" class="contact-link" target="_blank" rel="noopener">
      <span class="icon">${icon}</span>
      <span class="link-label">${label}<br/><small style="color:var(--muted);font-size:.75rem;">${sub}</small></span>
      <span class="link-arrow">→</span>
    </a>`
  ).join("");
}

// ── ANIMATIONS ───────────────────────────────────────────────────────────────

function initScrollReveal() {
  const items = document.querySelectorAll(".card, .project-item, .future-item, .contact-link");
  items.forEach(el => {
    el.style.opacity = "0";
    el.style.transform = "translateY(24px)";
    el.style.transition = "opacity .5s ease, transform .5s ease";
  });

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity = "1";
          e.target.style.transform = "translateY(0)";
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -40px 0px" }
  );

  items.forEach((el, i) => {
    setTimeout(() => observer.observe(el), i * 60);
  });
}

function initNavScroll() {
  const nav = document.querySelector("nav");
  window.addEventListener("scroll", () => {
    nav.style.background = window.scrollY > 40
      ? "rgba(2,4,9,0.92)"
      : "rgba(2,4,9,0.7)";
  });
}

function initOrbParallax() {
  const orb1 = document.querySelector(".orb-1");
  const orb2 = document.querySelector(".orb-2");
  window.addEventListener("mousemove", e => {
    const x = (e.clientX / window.innerWidth - 0.5) * 30;
    const y = (e.clientY / window.innerHeight - 0.5) * 30;
    if (orb1) orb1.style.transform = `translate(${x}px, ${y}px)`;
    if (orb2) orb2.style.transform = `translate(${-x}px, ${-y}px)`;
  });
}

function initTypewriter() {
  const subtitle = document.querySelector(".hero-subtitle");
  if (!subtitle) return;
  const text = subtitle.textContent;
  subtitle.textContent = "";
  let i = 0;
  const type = () => {
    if (i < text.length) {
      subtitle.textContent += text[i++];
      setTimeout(type, 40);
    }
  };
  setTimeout(type, 600);
}

// ── INIT ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  renderWhatIDo();
  renderProjects();
  renderFunProjects();
  renderFuture();
  renderContact();

  initScrollReveal();
  initNavScroll();
  initOrbParallax();
  initTypewriter();
});
