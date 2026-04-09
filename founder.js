// ── DATA ────────────────────────────────────────────────────────────────────

const WHAT_I_DO = [
  {
    icon: "⚡",
    title: "Software Development",
    text: "Building products from zero to one. Full-stack mindset — comfortable moving across layers to ship things that work.",
    tags: ["Full-Stack", "Product", "Shipping"],
  },
  {
    icon: "🤖",
    title: "AI & Automation",
    text: "Applying AI tools and automation to solve real problems faster. Exploring what's possible when you put the right tech in the right hands.",
    tags: ["AI", "LLMs", "Automation", "Tooling"],
  },
  {
    icon: "🚀",
    title: "Entrepreneurship",
    text: "Finding problems worth solving and building companies around them. Obsessed with the early stages — zero to traction.",
    tags: ["Startups", "Founder", "Strategy"],
  },
  {
    icon: "🌐",
    title: "Startup & Business Club — Uniandes",
    text: "Active member of the Startup and Business Club at Universidad de los Andes — connecting with founders, exploring ventures, and building within one of Latin America's top entrepreneurship ecosystems.",
    tags: ["Uniandes", "Startups", "Community", "Entrepreneurship"],
  },
];

const PROJECTS = [
  {
    num: "01",
    name: "Personal Vlog — Martin Ardila",
    desc: "This site. Documenting my journey as a founder and builder — honest, raw, and in real time.",
    status: "active",
    statusLabel: "Live",
  },
  {
    num: "02",
    name: "Startup Project [TBA]",
    desc: "An early-stage venture targeting a real gap in the market. More details coming soon. Alternatively, actively looking for an internship to generate real impact and experience — or a role at a startup to grow within the ecosystem.",
    status: "building",
    statusLabel: "Building",
  },
  {
    num: "03",
    name: "Spacetech IEEE Leadership",
    desc: "Developing and enjoying orbital mechanics and rocket science in community — leading a space technology chapter where curiosity meets rigorous engineering.",
    status: "active",
    statusLabel: "Live",
  },
  {
    num: "04",
    name: "Preicfes Orion",
    desc: "An enterprise offering pre-college prep courses across Colombia for families with low income — making quality education accessible to every corner of the country.",
    status: "building",
    statusLabel: "Building",
  },
];

const FUTURE_GOALS = [
  {
    icon: "🌎",
    title: "Developing ideas that drive real growth for LATAM",
    desc: "Focused on building ventures that address the region's most critical gaps — especially in hard tech and IoT, where Latin America has enormous untapped potential. The goal is to create solutions that are built here, for here, and eventually exported to the world.",
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
  renderFuture();
  renderContact();

  initScrollReveal();
  initNavScroll();
  initOrbParallax();
  initTypewriter();
});
